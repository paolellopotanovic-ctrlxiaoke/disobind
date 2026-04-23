"""
【中文解析-模块总览】
- 中心功能：create_input_embeddings.py 属于 Disobind 数据流水线脚本，用于数据抓取/清洗/映射/导出中的一个环节。
- 逻辑位置：该文件在“原始数据库 -> 统一样本定义 -> 训练/评估输入文件”链条中承担局部处理任务。
- 输入输出流水线：通常输入为数据库文件（csv/tsv/xlsx/json）、结构文件（pdb/cif）与序列标识（UniProt/PDB），输出为中间表、映射字典、模型可读输入（csv/h5/json/npy/fasta）。
- 可选项：多数脚本支持通过构造参数或命令行参数控制版本号、并行核数、路径和过滤阈值。
- 可视化能力：数据脚本本身通常不直接出图；如有图像分析一般在 analysis/ 目录统一完成。
- 数据格式与实验标签：核心监督标签通常是残基-残基接触图（contact map）或界面位点（interface residues）；本目录负责把外部异构数据规范化成论文实验使用的统一格式。
- 数据来源与论文逻辑：数据来源包含 DIBS/MFIB/DisProt/IDEAL/MobiDB/PDB/SIFTS 等；处理逻辑服务于“按二元复合体构建样本并对齐序列-结构关系”。
- 预训练扩展建议：若做进一步预训练，建议先扩展原始来源覆盖面、统一长度分桶策略、保留更多中间质控字段（置信度/覆盖率/冲突标记），再生成更大规模弱监督样本。
"""

import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import h5py
import time
import os
import json
from omegaconf import  OmegaConf
import torch
from torch import nn

from dataset.utility import get_embeddings, find_disorder_regions
from src.dataset_loaders import split_dataset, dataloader, create_residue_pairs


# 【中文解析-关键函数/类入口】以下对象是本文件的主执行入口，承担数据转换主链路。
class Embeddings():
	def __init__( self, scope = None, embedding_type = None, uniprot_seq = None,
					base_path = None, 
					fasta_file = None, emb_file = None,
					headers = None, load_cmap = True, eval_ = False ):
		"""
		Constructor
		"""
		# Seed for all PRNGs.
		self.seed = 1
		# Set seed for all PRNGs.
		self.seed_worker()
		# Version for the dataset directory.
		self.version = 21
		# self.imbl_sampler = None  # ["None", "smote", "adasyn"]  # Deprectaed
		# Whether to create global or local embeddings.
		self.scope = "global" if scope == None else scope
		# What embeddings to create [T5, ProstT5, ProSE, ESM2]
		self.embedding_type = "T5" if embedding_type == None else embedding_type
		# protT5/ProstT5 --> 1024; ProSE --> 6165; ESM2-650M --> 1280; ESM2-3B --> 2560; ESM2-15B --> 5120
		self.emb_size = 1024
		# Train:Dev:Test set partitions to be created.
		self.partitions = [0.9, 0.05, 0.05]
		# Max length for prot1/2.
		self.max_len = 100
		# If True, specifies path to get T5 embeddings at inference.
		self.eval = eval_

		# Dict to store prot1/2 embeddings with the entry_id as the key.
		self.p1_frag_emb = {}
		self.p2_frag_emb = {}
		self.logs = {}
		
		# self.no_neg_pairs = np.array( [] )
		# self.total_complexes = 0
		self.counts = {}
		self.crop_counts = [[], [], [], [], 0]

		self.headers = [] if headers == None else headers
		
		if self.scope != "global" and self.scope != "local":
			raise Exception( "Incorrect scope specified..." )

		if base_path == None:
			# Absolute path for the Database directory.
			self.base_path = os.path.abspath( f"../database/v_{self.version}" )
			self.dataset_path = f"{self.base_path}/{self.embedding_type}/{self.scope}-None/"

		else:
			self.base_path = base_path

		
		self.load_cmap = load_cmap
		if self.load_cmap:
			self.cmap_path = f"{self.base_path}/Target_bcmap_train_v_{self.version}.h5"
			self.cmaps = {}

		# .json file containing all the Uniprot seq.
		if uniprot_seq == None:
			# self.all_Uniprot_seq_file = "Disobind_Uniprot_seq"
			self.all_Uniprot_seq_file = f"{self.base_path}/Uniprot_seq.json"
			with open( self.all_Uniprot_seq_file, "r" ) as f:
				self.all_Uniprot_seq = json.load( f )
		else:
			self.all_Uniprot_seq = uniprot_seq

		# Fasta files.
		if fasta_file == None:
			self.fasta_file = f"{self.base_path}/{self.embedding_type}/train_fasta_{self.scope}-None_v_{self.version}.fasta"

			self.p1_p2_csv_file = f"{self.base_path}/prot_1-2_train_v_{self.version}.csv"
		
		else:
			self.fasta_file = fasta_file
				
		# File name for the generated embeddings.
		if emb_file == None:
			self.emb_file = f"{self.base_path}/{self.embedding_type}/train_emb_{self.scope}-None_v_{self.version}.h5"
		else:
			self.emb_file = emb_file
		

		self.fracton_positives_file = "fraction_positives.json"

		self.logger_file = "./Logs.yml"
		self.logs = {key:{} for key in ["Train", "Dev", "Test"]}



	def seed_worker( self ):
		"""
		Set the seed for all PRNGs for reproducibility.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		torch.manual_seed( self.seed )
		torch.cuda.manual_seed_all( self.seed )
		np.random.seed( self.seed )
		random.seed( self.seed )
	# 【中文解析-重点逻辑】forward 通常串联“读取输入 -> 清洗/映射 -> 导出中间/最终文件”，是该脚本最关键的数据流水线节点。
	def forward( self ):
		"""
		Create base_dir if not already existing.
		Obtain embeddings and create Train:Dev:Test sets.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		self.tic = time.time()

		dir_path = f"{self.base_path}/{self.embedding_type}/"
		if not os.path.exists( dir_path ):
			os.makedirs( dir_path )

		self.read_from_csv()
		self.initialize()
		
		if not os.path.exists( self.dataset_path ):
			os.makedirs( self.dataset_path )
		print( "-------------------------------------------" )
		os.chdir( self.dataset_path )

		train_keys, dev_keys, test_keys = self.split_dataset()

		print( "\nCreating train, dev and test sets" )
		self.create_input( train_keys, dev_keys, test_keys )

		self.create_summary_file( train_keys, dev_keys, test_keys )



	def read_from_csv( self ):
		"""
		Read from a csv file containing prot1 and prot2 info as:
			UniID1:start1:end1--UniID2:start2:end2_num,

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		print( "Loading prot1-prot2 headers csv file..." )
		with open( self.p1_p2_csv_file, "r" ) as f:
			self.headers = f.readlines()[0].split( "," )

		# If present, remove a null element due to extra "," at the end.
		# 	e.g. "A,B,C,D," --> ["A", "B", "C", "D", ""]
		if len(self.headers[-1]) == 0:
			self.headers = self.headers[:-1]
		print( len( self.headers ) )


	def create_fasta_from_headers( self ):
		"""
		Create fasta files for prot1 and prot2.
			e.g. >UniID1:start1:end1
				 Sequence
		This is the required input for creating embeddings.
		For scope == global --> create fasta file with complete Uniprot seq.
		For scope == local --> create fasta file with only the fragment Uniprot seq.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		print( "Creating FASTA file..." )
		w = open( self.fasta_file, "w" )
		self.flanked_uni_pos = {}
		considered = []

		for head in self.headers:
			head1, head2 = head.split( "--" )
			head2 = head2.split( "_" )[0]
			uni_id1, start1, end1 = head1.split( ":" )
			uni_id2, start2, end2 = head2.split( ":" )

			start1, end1 = int( start1 ), int( end1 )
			start2, end2 = int( start2 ), int( end2 )

			if self.scope == "global":
				if uni_id1 not in considered:
					w.writelines( f">{uni_id1}\n{self.all_Uniprot_seq[uni_id1]}\n\n" )
					considered.append( uni_id1 )
				
				if uni_id2 not in considered:
					w.writelines( f">{uni_id2}\n{self.all_Uniprot_seq[uni_id2]}\n\n" )
					considered.append( uni_id2 )
			
			elif self.scope == "local":
				if head1 not in considered:
					w.writelines( f">{head1}\n{self.all_Uniprot_seq[uni_id1][start1-1:end1]}\n\n" )
					considered.append( head1 )
				
				if head2 not in considered:
					w.writelines( f">{head2}\n{self.all_Uniprot_seq[uni_id2][start2-1:end2]}\n\n" )
					considered.append( head2 )

		w.close()



	def initialize( self, return_emb = False ):
		"""
		Input is a csv file containing Uni ID and residue ranges for prot1 and prot2.
			e.g. UniID1:start1:end1--UniID2:start2:end2_num
		Get the Uniprot IDs and residue ranges.
		Download the Uniprot sequences.
		Create fasta files.
		Get global or local embedding.

		Input:
		----------
		return_emb --> (bool) if True return the prot1/2 embeddings dict.

		Returns:
		----------
		p1_frag_emb --> (np.array) embedding for prot1.
		p2_frag_emb --> (np.array) embedding for prot2.
		"""
		self.create_fasta_from_headers()

		if not os.path.exists( self.emb_file ):
			print( "Creating prot1 embeddings..." )
			get_embeddings( self.embedding_type, self.fasta_file, self.emb_file, self.eval )

		else:
			print( "Embeddings for prot1/2 already exists..." )

		if self.scope == "global":
			self.get_global_embeddings()
		else:
			self.get_local_embeddings()

		if return_emb:
			return self.p1_frag_emb, self.p2_frag_emb



	def get_global_embeddings( self ):
		"""
		If scope == global -->  
				get embedding for entire Uniprot sequence.
				Slice out the embeddings for the specified residue range.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		print( "\nObtaining global embeddings.." )
		hf1 = h5py.File( self.emb_file, "r" )
		
		if self.load_cmap:
			hf2 = h5py.File( self.cmap_path, "r" )
		
		for head in self.headers:
			head1, head2 = head.split( "--" )
			head2 = head2.split( "_" )[0]
			uni_id1, start1, end1 = head1.split( ":" )
			uni_id2, start2, end2 = head2.split( ":" )

			emb1 = np.array( hf1[uni_id1], dtype = np.float16 )

			self.p1_frag_emb[head] = emb1[int( start1 )-1:int( end1 )]

			emb2 = np.array( hf1[uni_id2], dtype = np.float16 )
			self.p2_frag_emb[head] = emb2[int( start2 )-1:int( end2 )]

			if self.load_cmap:
				self.cmaps[head] = np.array( hf2[head], dtype = np.float16 )
		
		hf1.close()
		if self.load_cmap:
			hf2.close()



	def get_local_embeddings( self ):
		"""
		If scope == local
				obtain the fragment sequence specified by residue range.
				Get embedding for the fragment.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		
		hf1 = h5py.File( self.emb_file, "r" )
		if self.load_cmap:
			hf2 = h5py.File( self.cmap_path, "r" )

		for head in self.headers:
			head1, head2 = head.split( "--" )
			head2 = head2.split( "_" )[0]

			self.p1_frag_emb[head] = np.array( hf1[head1], dtype = np.float16 )
			self.p2_frag_emb[head] = np.array( hf1[head2], dtype = np.float16 )
			if self.load_cmap:
				self.cmaps[head] = np.array( hf2[head], dtype = np.float16 )
		hf1.close()
		if self.load_cmap:
			hf2.close()


	def split_dataset( self ):
		"""
		Split the dataset into a Train:Dev:Test set 
			of speciffed partitions.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		train_keys --> list of all entry_id's in Train set.
		dev_keys --> list of all entry_id's in Dev set.
		test_keys --> list of all entry_id's in Test set.
		"""
		train_keys, dev_keys, test_keys = split_dataset( list( self.p1_frag_emb.keys() ), self.partitions )

		return train_keys, dev_keys, test_keys



	def apply_padding( self, prot1, prot2, cmap, key ):
		"""
		Pad the prot1/2 and contact map upto max len.
			To do so, create a 0's array for the dimension
			(max_len x C) and (max_len x max_len) for prot1/2 and cmap respectively.
			To this add the original (L1 x C)/(L2 x C) or (L1 x L2) array.
		Also create a target_mask for cmap with 1's upto (L1 x L2) and rest 0's

		Input:
		----------
		prot1 --> (np.array) prot1 embeddings (L1 x C).
		prot2 --> (np.array) prot2 embeddings (L2 x C).
			L1,L2 --> length of pro1/2; C --> embedding size.
		cmap --> (np.array) contact map for prot1 and prot2 (L1 x L2).

		Returns:
		----------
		p1_padded --> (np.array) prot1 embedding padded upto max_len along L1.
		p2_padded --> (np.array) prot2 embedding padded upto max_len along L2.
		target_padded --> (np.array) contact map padded upto max_len along L1 and L2.
		target_mask --> (np.array) a binary mask with 1's upto (L1 x L2).
		"""
		res1 = prot1.shape[0]
		res2 = prot2.shape[0]

		p1_padded = np.zeros( ( self.max_len, self.emb_size ) )
		p1_padded[:res1,:] = prot1

		p2_padded = np.zeros( ( self.max_len, self.emb_size ) )
		p2_padded[:res2,:] = prot2

		target_mask = np.zeros( ( self.max_len, self.max_len ) )
		target_mask[:res1,:res2] = 1
		target_padded = np.copy( target_mask )
		
		target_padded[:res1,:res2] = cmap

		return p1_padded, p2_padded, target_padded, target_mask



	def create_input( self, train_keys, dev_keys, test_keys ):
		"""
		Get the embeddings for prot1/2 and cmap for Train:Dev:Test sets.
		Get concatenated arrays for all entries for prot1/2 and cmap.
		Count the no. of 0's and 1's for each set.
		Save the arrays on disk for eachset.

		Input:
		----------
		train_keys --> list of all entry_id's in Train set.
		dev_keys --> list of all entry_id's in Dev set.
		test_keys --> list of all entry_id's in Test set.

		Returns:
		----------
		None
		"""
		if os.path.exists( self.logger_file ):
			self.logs = OmegaConf.load( self.logger_file )
		
		for set_, label in zip( [dev_keys, test_keys, train_keys], ["Dev", "Test", "Train"] ):
			print( f"\n{label} set.........\n" )
			# [contacts, non_contacts, prot1_length, prot2_length]
			self.counts[label] = [[], [], [], []]

			# Do not recreate the dataset it already exists.
			if os.path.exists( f"./{label}_set_{self.scope}_v_{self.version}.npy" ):
				print( f"{label} dataset already exists..." )

			else:
				data = []

				self.train = True if label == "Train" else False
				prot1, prot2, target = [], [], []

				idx = 0
				total_entries = len( set_ )
				pos, neg = 0, 0
				lengths = np.array( [] )
				for key in set_:

					# These merged entries have a problematic UniProt residue position.
					if key in ["Q01468:2:63--Q01468:63:62_2", "Q01468:2:63--Q01468:63:62_3"]:
						continue

					print( f"{idx}/{total_entries} :: {label} set --> {sum( self.counts[label][0] )} \t {sum( self.counts[label][1] )} {key}" )
					p1_emb = self.p1_frag_emb[key]
					p2_emb = self.p2_frag_emb[key]
					cmap = self.cmaps[key]
					p = np.count_nonzero( cmap )
					self.counts[label][0].append( p )
					self.counts[label][1].append( cmap.size - p )
					self.counts[label][2].append( p1_emb.shape[0] )
					self.counts[label][3].append( p2_emb.shape[0] )

					# if self.train and self.create_crops:
					# 	p1_crops, p2_crops, cmap_crops = self.get_crops( p1_emb, p2_emb, cmap )

					# 	prot1.extend(  p1_crops )
					# 	prot2.extend(  p2_crops )
					# 	target.extend( cmap_crops )

					# else:
					p1_padded, p2_padded, target_padded, target_mask = self.apply_padding( p1_emb, p2_emb, cmap, key )

					prot1.append(  p1_padded )
					prot2.append(  p2_padded )
					target.append( np.concatenate( ( target_padded, target_mask ), axis = 1 ) )


					self.p1_frag_emb.pop( key )
					self.p2_frag_emb.pop( key )
					self.cmaps.pop( key )
					idx += 1
					print( f"{idx}/{total_entries} :: {label} set --> {sum( self.counts[label][0] )} \t {sum( self.counts[label][1] )} {key}" )

				self.clean_up( prot1, prot2, target, label )



	def create_plot( self, label, crop ):
		"""
		Plot distributions of the contact count and prot1/2 lengths
			for cropped and uncropped datasets.

		Input:
		----------
		label --> (str) label for the plot to be saved.
		crop --> (cool) to segreagte between cropped and uncropped dataset.

		Returns:
		----------
		None
		"""
		if crop:
			contacts, len1, len2 = self.crop_counts[0], self.crop_counts[2],self.crop_counts[3]
		else:
			contacts, len1, len2 = self.counts[label][0], self.counts[label][2],self.counts[label][3]

		
		fig, axis = plt.subplots( 1, 3, figsize = ( 20, 15 ) )
		axis[0].hist( contacts, bins = 20 )
		axis[1].hist( len1, bins = 20 )
		axis[2].hist( len2, bins = 20 )
		axis[0].xaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[1].xaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[2].xaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[0].yaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[1].yaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[2].yaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[0].set_ylabel( "Counts", fontsize = 16 )
		axis[1].set_ylabel( "Counts", fontsize = 16 )
		axis[2].set_ylabel( "Counts", fontsize = 16 )
		axis[0].set_title( "Contacts distribution", fontsize = 16 )
		axis[1].set_title( "Prot1 lengths distribution", fontsize = 16 )
		axis[2].set_title( "Prot2 lengths distribution", fontsize = 16 )

		if crop:
			plt.savefig( f"{self.dataset_path}{label}_Summary_plots_cropped.png", dpi = 300 )
		else:
			plt.savefig( f"{self.dataset_path}{label}_Summary_plots.png", dpi = 300 )
		plt.close()


	def clean_up( self, prot1, prot2, target, label ):
		"""
		Save the prot1/2 and cmap arrays for the Train/Dev/Test set on disk.
		Concatenate the prot1/2 embeddings and contact maps for all entries in each set.
			This results in a 3D array of dimension:
			prot1 --> [N, L1, C]
			prot2 --> [N, L2, C]
			cmap --> [N, L1, L2]
				L1 = L2 due to padding.
		Also concatenate the 3D arrays for prto1/2 and cmap together along axis = 2.
			e.g. [N, L1, C], [N, L2, C], [N, L1, L2] --> [N, L1, C+C+L2]

		Input:
		----------
		prot1 --> list of arrays for prot1 embeddings Nx[L1, C].
		prot1 --> list of arrays for prot2 embeddings Nx[L2, C].
		target --> list of cmap arrays Nx[L1, L2].
		label --> (str) to specify Train/Dev/Test set.

		Returns:
		----------
		None
		"""
		self.create_plot( label, crop = False )
		if self.train:
			self.create_plot( label, crop = True )

		prot1 = np.stack( prot1 )
		prot2 = np.stack( prot2 )
		target = np.stack( target )
		data = np.concatenate( ( prot1, prot2, target ), axis = 2 )

		self.logs[label]["Total_pairs"] = sum( self.counts[label][0] ) + sum( self.counts[label][1] )
		self.logs[label]["Contact_pairs"] = sum( self.counts[label][0] )
		self.logs[label]["Non_contact_pairs"] = sum( self.counts[label][1] )
		# if self.create_crops:
			# self.logs[label]["Cropped_dataset"] = [sum( self.crop_counts[0] ), sum( self.crop_counts[1] )]
		self.write_output( label, data )

		del data



	def write_output( self, label, data ):
		"""
		Save the Train/Dev/Test dataset created on disk.

		Input:
		----------
		label --> (str) to specify Train/Dev/Test set.
		data --> concatenated array for the Train/Dev/Test set 
				of dimension [N, L1, C+C+L2]

		Returns:
		----------
		None
		"""
		print( f"Writing output files for {label} set..." )
		np.save( f"./{label}_set_{self.scope}_v_{self.version}.npy",
					data, allow_pickle = True )

		OmegaConf.save( config = self.logs, f = self.logger_file )

		del data


	def get_fraction_of_positives( self, train_keys ):
		"""
		Get the fraction of positives across all entries in train set, for all coarse grainings
			for interaction and interface objectives.
		Required for random baseline prediction.

		Input:
		----------
		train_keys --> list of all entry_id's in Train set.

		Returns:
		----------
		frac_pos --> dict containng fraction of positives in train set, for all coarse grainings 
					for interaction and interface objectives.
		"""
		frac_pos = {}

		hf = h5py.File( self.cmap_path, "r" )

		for obj in ["interaction", "interface"]:
			frac_pos[obj] = {}
			for cg in [1, 5, 10]:
				positives, total_res = 0, 0

				for key in train_keys:
					cmap = np.array( hf[key] )

					p1_res_num, p2_res_num = cmap.shape

					pad = np.zeros( ( self.max_len, self.max_len ) )
					pad[:p1_res_num,:p2_res_num] = cmap
					target_padded = pad

					m = nn.MaxPool2d( kernel_size = cg, stride = cg ).to( "cuda" )
					cg_target = m( torch.from_numpy( target_padded ).unsqueeze( 0 ).to( "cuda" ) )
					cg_target = cg_target.squeeze( 0 ).detach().cpu().numpy()
					# H_out = ( ( H_in + 2*padding - kernel_size )/stride ) + 1
					# 		Here, padding = 0 here; kernel_size = stride = cg.
					cg_p1_res_num = math.ceil( ( ( p1_res_num - cg )/cg ) + 1 )
					cg_p2_res_num = math.ceil( ( ( p2_res_num - cg )/cg ) + 1 )

					if obj == "interaction":
						total_res += cg_p1_res_num*cg_p2_res_num
						positives += np.count_nonzero( cg_target )

					else:
						total_res += cg_p1_res_num+cg_p2_res_num

						idx = np.where( cg_target == 1 )
						positives += len( np.unique( idx[0] ) ) + len( np.unique( idx[1] ) )

				frac_pos[obj][cg] = np.round( positives/total_res, 4 )
		hf.close()

		with open( self.fracton_positives_file, "w" ) as w:
			json.dump( frac_pos, w )
		return frac_pos



	def create_summary_file( self, train_keys, dev_keys, test_keys ):
		"""
		Save the logs in a txt file.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		OmegaConf.save( config = self.counts, f = "Counts_full_dataset" )

		frac_pos = frac_pos = self.get_fraction_of_positives( train_keys )

		self.toc = time.time()
		time_ = ( self.toc - self.tic )
		self.logs["time_taken"] = time_

		with open( f"./Summary_res_emb_{self.scope}_{self.version}.txt", "w" ) as w:
			w.writelines( "-----------------------------------------\n" )
			w.writelines( f"Total time taken = {self.logs['time_taken']} seconds\n" )
			w.writelines( f"Partitions --> {self.partitions[0]}: {self.partitions[1]}: {self.partitions[2]}\n" )
			w.writelines( f"Merged complexes --> {len( train_keys )}: {len( dev_keys )}: {len( test_keys )}\n" )

			all_res_pairs = 0

			for key1 in self.logs.keys():
				if key1 != "time_taken":
					w.writelines( f"{key1}------------------------\n" )
					all_res_pairs += self.logs[key1]["Total_pairs"]
					for key2 in self.logs[key1].keys():
						w.writelines( f"{key2}: {self.logs[key1][key2]}\n" )
					w.writelines( "\n\n" )
			w.writelines( f"Total residue pairs = {all_res_pairs}\n" )

			w.writelines( "\n---------------------------------------\n" )
			for obj in frac_pos.keys():
				w.writelines( f"{obj}\n" )
				for cg in frac_pos[obj].keys():
					w.writelines( f"{cg} = {frac_pos[obj][cg]}\t" )
				w.writelines( "\n" )



################################################
if __name__ == "__main__":
	Embeddings().forward()
	print( "May the Force be with you..." )
################################################

"""
【中文解析-可演化改动建议】
1) 将硬编码路径迁移到统一配置文件（YAML/ENV）并加入路径合法性校验。
2) 为每个中间文件增加 schema 校验（字段名/类型/范围），降低跨脚本耦合导致的隐式错误。
3) 增加小样本 smoke test（例如 3~5 条记录）和数据快照测试，保障重构时行为一致。
4) 若面向预训练，建议输出 token 级别元信息（来源数据库、结构置信度、对齐覆盖率）便于构建多任务标签。
"""
