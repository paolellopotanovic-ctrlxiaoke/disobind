"""
【中文解析-模块总览】
- 中心功能：3_create_merged_binary_complexes.py 属于 Disobind 数据流水线脚本，用于数据抓取/清洗/映射/导出中的一个环节。
- 逻辑位置：该文件在“原始数据库 -> 统一样本定义 -> 训练/评估输入文件”链条中承担局部处理任务。
- 输入输出流水线：通常输入为数据库文件（csv/tsv/xlsx/json）、结构文件（pdb/cif）与序列标识（UniProt/PDB），输出为中间表、映射字典、模型可读输入（csv/h5/json/npy/fasta）。
- 可选项：多数脚本支持通过构造参数或命令行参数控制版本号、并行核数、路径和过滤阈值。
- 可视化能力：数据脚本本身通常不直接出图；如有图像分析一般在 analysis/ 目录统一完成。
- 数据格式与实验标签：核心监督标签通常是残基-残基接触图（contact map）或界面位点（interface residues）；本目录负责把外部异构数据规范化成论文实验使用的统一格式。
- 数据来源与论文逻辑：数据来源包含 DIBS/MFIB/DisProt/IDEAL/MobiDB/PDB/SIFTS 等；处理逻辑服务于“按二元复合体构建样本并对齐序列-结构关系”。
- 预训练扩展建议：若做进一步预训练，建议先扩展原始来源覆盖面、统一长度分桶策略、保留更多中间质控字段（置信度/覆盖率/冲突标记），再生成更大规模弱监督样本。
"""

############## Creating merged binary complexes.##############
###------> P.S. Holy Grail for the project ###################
####### ------>"May the Force serve u well..." <------########
##############################################################


############# One above all #############
##-------------------------------------##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import argparse
import subprocess
import glob
from omegaconf import OmegaConf
import time
import h5py
from multiprocessing import Pool
from functools import partial
import tqdm
import json

from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from utility import ( load_PDB, get_coordinates, get_contact_map,
					sort_by_residue_positions,
					check_for_overlap, merge_residue_positions,
					merged_seq_exceeds_maxlen )

import warnings
warnings.filterwarnings("ignore")

np.random.seed( 11 )

# 【中文解析-关键函数/类入口】以下对象是本文件的主执行入口，承担数据转换主链路。
class Dataset ():
	def __init__ ( self, version, cores ):
		"""
		Constructor
		"""
		self.tim = time.time()   # Just for calculating total time.
		self.version = version
		self.base_path = f"../database/v_{self.version}"
		self.dataset_version = "200_20_0.2"
		# Dir containing PDB/CIF files.
		self.pdb_path = "../Combined_PDBs/"
		# Name of file storing Uniprot sequences.
		self.uni_seq_file_name = "Uniprot_seq"
		self.uni_seq_path = f"../Disobind_dataset_{self.dataset_version}/{self.uni_seq_file_name}.json"
		# Dir storing binary complexes created by previous script.
		self.binary_complexes_dir = f"../Disobind_dataset_{self.dataset_version}/Binary_complexes_None/"
		# File containing keys for binary complexes created by previous script.
		self.binary_complexes_file = f"../Disobind_dataset_{self.dataset_version}/Binary_complexes_None.txt"
		# Dir to store the valid binary complexes.
		self.valid_binary_complexes_dir = "./Valid_Binary_Complexes/"
		# Dir to store the logs for obtaiing valid binary complexes.
		self.valid_binary_complexes_logs_dir = "./Valid_Binary_Complexes_logs/"
		# File to store keys for the valid binary complexes.
		self.valid_binary_complexes_file = "./valid_Binary_Complexes.txt"
		# Dir to store overlapping sets of binary complexes.
		self.overlapping_uni_pairs_dir = "./overlapping_Uni_pairs/"
		# File to store keys for overlapping sets of binary complexes.
		self.overlapping_uni_pairs_file = "./overlapping_uni_pairs.txt"
		# Dir to store the merged binary complexes.
		self.merged_binary_complexes_dir = "./merged_binary_complexes/"
		# Dir to store the logs for obtaiing merged binary complexes.
		self.merged_binary_complexes_logs_dir = "./merged_binary_complexes_logs/"
		# File to store keys for merged binary complexes.
		self.merged_binary_complexes_file = "./merged_binary_complexes.txt"

		self.dtype_dict = {"PDB ID": str, "Asym ID1": str, "Asym ID2": str,
					"Auth Asym ID1": str, "Auth Asym ID2": str, 
					"Uniprot accession1": str, "Uniprot accession2": str,
					"Uniprot positions1": np.int16, "Uniprot positions2": np.int16,
					"PDB positions1": np.int16, "PDB positions2": np.int16}
		# Max length of protein seq.
		self.max_len = 100
		# Min length of protein seq.
		self.min_len = 20
		# Size of the first batch - module1.
		self.m1_first_batch  = 11000
		# Size of the all subsequent batches - module1.
		self.m1_batch_size = 50
		# Batch size - module3.
		self.m3_batch_size = 1500
		# threshold for defining a contact.
		self.contact_threshold = 8
		# No. of cores to parallelize the tasks.
		self.cores = cores
		# If True consider all models in a PDB else consider only the 1st model.
		self.all_models = True
		# If True consider only 1 conformer for each Uniprot ID pair.
		self.no_hetero = False

		self.logger_file = f"./Logs_v_{self.version}.json"
		self.logger = {"time_taken": {}, "counts": {}}
	# 【中文解析-重点逻辑】forward 通常串联“读取输入 -> 清洗/映射 -> 导出中间/最终文件”，是该脚本最关键的数据流水线节点。
	def forward( self ):
		"""	
		Create the base directory for storing the required files 
				if not already existing.\
		Execute all modules sequentially.
		"""
		if not os.path.exists( self.base_path ):
			os.makedirs( f"{self.base_path}/" )
		else:
			print( f"{self.version} directory already exists." )
		
		# Move to the base directory.
		os.chdir( f"{self.base_path}/" )

		# <=======================================================> #
		"""
		Given the binary complexes for each Uniprot ID pair created by the previous script,
			remove those which do not pass the checks as specified in module1().
		"""
		if os.path.exists( self.logger_file ):
			# Load the logger state.
			with open( self.logger_file, "r" ) as f:
				self.logger = json.load( f )
		
		if os.path.exists( self.valid_binary_complexes_file ):
			print( "Using pre-created valid Uniprot ID pairs...\n" )
			print( "Total binary complexes obtained = ", self.logger["counts"]["total_binary_complexes"] )
			print( "Valid Uniprot ID pairs obtained = ", self.logger["counts"]["valid_uniprot_ID_pairs"] )
			print( "Valid binary complexes obtained = ", self.logger["counts"]["valid_binary_complexes"] )

		else:
			tic = time.time()
			print( "Creating valid Uniprot ID pairs...\n" )
			for key in ["excess_res_coords", f"0_contacts", "missing_chain_in_model"]:
				self.logger[key] = [[], 0]

			self.module1()
			toc = time.time()
			time1 = toc - tic
			print( f"Time taken to obtain valid binary complexes = {time1/3600} hours" )
			self.logger["time_taken"]["valid_binary_complexes"] = time1

			print( "Total binary complexes obtained = ", self.logger["counts"]["total_binary_complexes"] )
			print( "Valid Uniprot ID pairs obtained = ", self.logger["counts"]["valid_uniprot_ID_pairs"] )
			print( "Valid binary complexes obtained = ", self.logger["counts"]["valid_binary_complexes"] )
			print( "Time taken to create valid binary complexes = ", time1 )

			# Save the state.
			with open( self.logger_file, "w" ) as w:
				json.dump( self.logger, w )

		print( "\n------------------------------------------------------------------\n" )
		# <=======================================================> #
		"""
		For all binary complexes of each Uniprot ID pair, 
			sort and split them into non-overlapping sets such that
			each set contains all overlapping binary complexes only.
		"""
		if os.path.exists( self.overlapping_uni_pairs_file ):
			print( "Using pre-created overlapping sets..." )
			# # Load the logger state.
			# with open( self.logger_file, "r" ) as f:
			# 	self.logger = json.load( f )

			print( "Non-overlapping Uniprot ID pairs obtained = ", self.logger["counts"]["non_overlapping_set_creation"] )
		
		else:
			tic = time.time()
			print( "Creating overlapping sets of binary complexes...\n" )
			self.module2()
			toc = time.time()
			time2 = toc - tic
			print( f"Time taken to create overlapping sets = {time2/60} minutes" )
			self.logger["time_taken"]["non_overlapping_set_creation"] = time2
			print( "Non-overlapping Uniprot ID pairs obtained = ", self.logger["counts"]["non_overlapping_set_creation"] )

			# Save the state.
			with open( self.logger_file, "w" ) as w:
				json.dump( self.logger, w )

		print( "\n------------------------------------------------------------------\n" )
		# <=======================================================> #
		"""
		Create contact maps and merge all overlapping binary 
			complexes for each Uniprot ID pair.
		"""
		print( "Loading Uniprot sequences..\n" )
		with open( self.uni_seq_path, "r" ) as f:
			self.Uniprot_seq_dict = json.load( f )
		
		if os.path.exists( self.merged_binary_complexes_file ):
			print( "Using pre-merged binary complexes..." )
			# # Load the logger state.
			# with open( self.logger_file, "r" ) as f:
			# 	self.logger = json.load( f )

			print( "Merged binary complexes  obtained = ", self.logger["counts"]["merged_binary_complexes"] )
		
		else:
			tic = time.time()
			for key in ["uni_seq_pos_mismatch", 
						"mismatch_cmap_seq_dim", 
						"mismatch_prot1/prot2_length", "no_overlap_prot2_uni_pos", 
						"merged_len_exceed", "hetero"]:
				self.logger[key] = [[], 0]
			
			print( "Creating merged binary complexes...\n" )
			self.module3()
			toc = time.time()
			time3 = toc - tic
			print( f"Time taken to create merged binary complexes = {time3/60} minutes" )
			self.logger["time_taken"]["merged_binary_complexes"] = time3
			print( "Merged binary complexes  obtained = ", self.logger["counts"]["merged_binary_complexes"] )
			
			# Save the state.
			with open( self.logger_file, "w" ) as w:
				json.dump( self.logger, w )


		print( "\n------------------------------------------------------------------\n" )
		# <=======================================================> #
		"""
		Saving the logs.
		Also saving Uniprot seq file in the base directory.
		"""
		with open( f"./{self.uni_seq_file_name}.json", "w" ) as w:
			json.dump( self.Uniprot_seq_dict, w )
		
		self.save_logs()



###################################################################################################################
###################################################################################################################
###################################################################################################################
	def accio_uni_seq( self, uni_id, uni_pos ):
		"""
		Obtain the residues from the Uniprot sequence for the 
			specified Uniprot residue positions.
		List indexing starts from 0 while residue position in seq starts from 1.
		     [0 1 2 3 4 5 6 7 8 9]  <- List indexing
		     [1 2 3 4 5 6 7 8 9 10] <- Uniprot positions
		e.g. [A B C D E F G H I J]
				Uniprot residues 3-7 = CDEFG
				List positions = 2-6 --> [uni_pos-1]

		Input:
		----------
		uni_id --> (str) UniProt ID.
		uni_pos --> list of Uniprot residue positions.

		Returns:
		----------
		seq --> (str) seq for the given uniprot ID for the required residues.
		"""
		if uni_id in self.Uniprot_seq_dict.keys():
			uni_seq = self.Uniprot_seq_dict[uni_id]

			if uni_pos[-1] > len( uni_seq ):
				return 0

			else:
				seq = [uni_seq[pos-1] for pos in uni_pos]
				return seq
		else:
			return None


	def create_PDBs_dict( self, pdb_ids ):
		"""
		Create a dict for storing all required PDBs on memory.

		Input:
		----------
		pdb_ids_list --> list of PDB IDs.

		Returns:
		----------
		pdbs_dict --> nested dict storing PDB ID as keys 
				and the required chains for all models.
		pdbs_dict{
				pdb: models
		}
		"""
		pdbs_dict = {}
		for i, pdb in enumerate( set( pdb_ids ) ):
			pdbs_dict[pdb] = load_PDB( pdb, self.pdb_path )
		return pdbs_dict



	def create_contact_map( self, models, prot1_pdb_pos, prot2_pdb_pos, 
							prot1_chain, prot2_chain, all_models ):
		"""
		Create a contact map for the given sequences considering contacts from all the
		 	models in the PDB. 
		 	Multi model PDBs may arise from structures solved via NMR, SAXS, MD, etc.

		Input:
		----------
		models --> all models obtained from the PDB/CIF file.
		prot1_pdb_pos --> (list) mapped pdb pos for protein1.
		prot2_pdb_pos --> (list) mapped pdb pos for protein2.
		prot1_chain --> (str) chain ID(Asym/Auth Asym) for protein 1 in the PDB.
		prot2_chain --> (str) chain ID(Asym/Auth Asym) for protein 2 in the PDB.
		all_models --> (bool) If True consider all models for contact map
					 creation else only the first.

		Returns:
		----------
		missing_chain --> (bool) True if any chain in any model fo PDB id missing else False.
		excess_res_coords --> (bool) True if no. of coords selected do not match no. of residues.
		num_conformers --> (int) no. of models from PDB that have been considered.
		agg_contact_map --> (np.array) binary comntact map containing contacts 
						across all models in the PDB.
		"""
		# Create a contact map of dimension [L1,L2]
		# 	where L1 --> prot1 length; L2 --> prot2 length.
		dim = ( len( prot1_pdb_pos ), len( prot2_pdb_pos ) )
		agg_contact_map = np.zeros( dim, dtype = np.int8 )

		missing_chain, excess_res_coords = False, False
		# Keep track of the no. of conformers in the PDB.
		num_conformers = 0
		
		for model in models:
			# Loop over all the models in the PDB
			# Get coords from PDB.
			# Can happen if the model has a missing chain.
			
			# Check if the model contains both the required chain IDs.
			chain_ids = [chain.id for chain in model.get_chains()]
			if prot1_chain not in chain_ids:
				missing_chain = True
				continue
			elif prot2_chain not in chain_ids:
				missing_chain = True
				continue

			coords1 = get_coordinates( model[prot1_chain], prot1_pdb_pos )
			coords2 = get_coordinates( model[prot2_chain], prot2_pdb_pos )

			# Igone if the chain ID exists, but no coordinates are present.
			if len( coords1 ) == 0 or len( coords2 ) == 0:
				# print( "Skipped model due to one of the chains missing..." )
				missing_chain = True
				continue

			# In case PDB contains duplicated resuidues (e.g. 50, 51, 52, 52, 53, 54),
			# 		no. of coordinates selected can be more than no. of residues.
			# Could be due to a missing residue or missing CA for some residues.
			# Also if some residues are marked as HETATM in PDB.
			# Ignore such PDBs.
			if len( coords1 ) != len( prot1_pdb_pos ) or len( coords2 ) != len( prot2_pdb_pos ):
				excess_res_coords = True
				break

			contact_map = get_contact_map( coords1, coords2, self.contact_threshold )
			
			# Aggregate contacts from all the models
			# agg_contact_map = np.logical_or( agg_contact_map, contact_map )
			agg_contact_map = agg_contact_map + contact_map

			num_conformers += 1
			if not all_models:
				break

		del coords1, coords2, models, prot1_pdb_pos, prot2_pdb_pos
		return missing_chain, excess_res_coords, num_conformers, agg_contact_map



###################################################################################################################
###################################################################################################################
###################################################################################################################
	def check_for_valid_PDBs( self, upid ):
		"""
		Parse through all the binary complexes for a given Uniprot ID pair 
				stored in respective .json  files.
				identified by "Uniprot_ID1--Uniprot_ID2".
				This forms the upid.
		Here we need to filter out binary complexes:
			That have 0 contacts in the contact maps.
			Entries for whcih PDB have more/less coordinates than required.
		For PDBs having a missing chain for any model.
			we only ignore the model not the entire PDB.
		Entries which pass all the checks are considered valid_binary_complexes.

		Input:
		----------
		upid --> (str) UniProt ID pair e.g. "{uni_id1}--{uni_id2}_{copy_num}".

		Returns:
		----------
		upid --> (str) Uniprot ID pair, same as input.
		count_binary_complexes --> (int) no. of valid binary complexes 
				selected for the upid.
		logs_dict --> dict for logging required info. Include:
				excess_res_coords --> mismatch in no. of coordinates obtained 
							and the no pdb of residues.
				0_contacts --> binary complexes with 0 contacts in cmap.
				missing_chain_in_model --> complexes for which a model has a missing chain.
		"""
		logs_dict = {i:0 for i in ["excess_res_coords", "0_contacts",
									"missing_chain_in_model"]}

		count_binary_complexes = 0
		valid_df = pd.DataFrame()

		df = pd.read_json( f"{self.binary_complexes_dir}{upid}.json", dtype = self.dtype_dict )
		total_binary_complexes = len( df )

		# If an entry has been processed, its logs file would be present else not.
		if not os.path.exists( f"{self.valid_binary_complexes_logs_dir}{upid}.json" ):
			groups = df.groupby( by = "PDB ID" )
			del df

			for grp in groups:
				data = grp[1]

				pdb = grp[0]
				models = load_PDB( pdb, self.pdb_path )
				all_indexes = data.index
				for i in all_indexes:
					if len( data["PDB positions1"][i] ) > self.max_len or len( data["PDB positions2"][i] ) > self.max_len:
						print( upid, "  ", pdb, "  ", data["Auth Asym ID1"][i] )
						print( len( data["PDB positions1"][i] ), "  ", len( data["PDB positions2"][i] ) )
						raise Exception( "Chains longer than max_len...\n" )


					#### Get the aggregate contact_map.
					####------------------------------------------------------
					chain_missing, excess_res_coords, num_conformers, agg_cmap  = self.create_contact_map(
																							models,
																							data["PDB positions1"][i], 
																							data["PDB positions2"][i],
																							data["Auth Asym ID1"][i], 
																							data["Auth Asym ID2"][i],
																							all_models = False )

					# Remove if the no. of coordinates are more/less than required.
					if excess_res_coords:
						data = data.drop( i )
						logs_dict[f"excess_res_coords"] += 1
						continue

					# Remove if an entry if there are no contacts..
					elif np.count_nonzero( agg_cmap ) == 0:
						data = data.drop( i )
						logs_dict[f"0_contacts"] += 1
						continue

					if chain_missing:
						logs_dict[f"missing_chain_in_model"] += 1

				del models
				valid_df = pd.concat( [valid_df, data] )
				del data

			upid_ = upid
			if len( valid_df ) == 0:
				upid = None
				count_binary_complexes = 0
			
			else:
				valid_df = valid_df.reset_index()
				valid_df.to_hdf( f"{self.valid_binary_complexes_dir}{upid_}.h5", key = "data", mode = "w" )
				count_binary_complexes = len( valid_df )
				del valid_df

			with open( f"{self.valid_binary_complexes_logs_dir}{upid_}.json", "w" ) as w:
				json.dump( logs_dict, w )
		
		# Load if the upid has already been processed.
		else:
			with open( f"{self.valid_binary_complexes_logs_dir}{upid}.json", "r" ) as f:
				logs_dict = json.load( f )
			
			# A h5 file will be saved only if the upid had at least 1 valid binary complex else not.
			if os.path.exists( f"{self.valid_binary_complexes_dir}{upid}.h5" ):
				valid_df = pd.read_hdf( f"{self.valid_binary_complexes_dir}{upid}.h5", dtype = self.dtype_dict )
				count_binary_complexes = len( valid_df )
			else:
				count_binary_complexes = 0
				upid = None


		return [upid, total_binary_complexes, count_binary_complexes, logs_dict]



	def module1( self ):
		#### Module1: Parallelize selecting valid binary complexes.
		##-----------------------------------------------------------------##
		"""
		Select valid binary complexes across all Uniprot ID pairs.
			See self.check_for_valid_PDBs() for criterion.
		For memory efficiency:
			Parallelize batches of Uniprot ID pairs instead of all together.
		Note the time taken.

		**Note
		Loading large CIF files in parallel causes significantly huge memory consumption.
			e.g. 8glv, 8oj8, 8j07, 8qo9, etc.
		Most of the Uniprot ID pairs containing such large CIF files are horded towards the end.
		Thus we process them in smaller batches that can be loaded in memory.
		Loading and processing them serially will take a large amount of time.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		# Create a dir to save all valid binary complexes 
		# 		if not already existing.
		count_total_binary_complexes, count_valid_binary_complexes = 0, 0
		if not os.path.exists( self.valid_binary_complexes_dir ):
			os.makedirs( self.valid_binary_complexes_dir )
		if not os.path.exists( self.valid_binary_complexes_logs_dir ):
			os.makedirs( self.valid_binary_complexes_logs_dir )

		with open( self.binary_complexes_file ) as f:
			binary_complexes = f.readlines()[0].split( "," )
		total_complexes = len( binary_complexes )
			# del binary_complexes
		print( f"Total Uniprot ID pairs = {total_complexes}\n" )

		valid_uni_pairs = []

		# Processing batches in parallel.
		# Creating a major first batch and subsequent small batches 
		# 			to efficiently handle the large CIF containing entries.
		batches = np.arange( self.m1_first_batch, len( binary_complexes ), self.m1_batch_size )
		start = 0
		for end in batches:
			batch = binary_complexes[start:end]

			with Pool( cores ) as p:
				for result in tqdm.tqdm( p.imap_unordered( self.check_for_valid_PDBs, 
															batch ), total = len( batch ) ):

					upid, total_count, valid_count, logs_dict = result

					for key in logs_dict.keys():
						self.logger[key][1] += logs_dict[key]
					
					count_total_binary_complexes += total_count
					if upid != None:
						count_valid_binary_complexes += valid_count
						valid_uni_pairs.append( upid )
			print( f"Completed for batch {start}-{end}..." )
			start = end

		with open( self.valid_binary_complexes_file, "w" ) as w:
			w.writelines( ",".join( valid_uni_pairs ) )
		self.logger["counts"]["valid_uniprot_ID_pairs"] = len( valid_uni_pairs )
		self.logger["counts"]["valid_binary_complexes"] = count_valid_binary_complexes
		self.logger["counts"]["total_binary_complexes"] = count_total_binary_complexes



###################################################################################################################
###################################################################################################################
###################################################################################################################
	def save_overlapping_entries( self, df, new_df_name, last_checkpoint, idx = None ):
		"""
		Save each set of overlapping entries as a csv file.
			Delete the saved entries -- clears memory.
		
		Input:
		----------
		df --> dataframe storing all entries per uniprot ID pair.
		new_df_name --> name for the csv file to be saved.
		last_checkpoint --> start point to consider entries to be saved.
		idx --> end point to consider entries to be saved.

		Returns:
		----------
		df --> pandas dataframe
				for efficiency, rows that were saved are removed.
		"""
		# For only a single entry.
		if idx == None:
			df_new = df.iloc[[last_checkpoint],:].copy()
			df = df.drop( df.index[last_checkpoint] )

		else:
			df_new = df.iloc[last_checkpoint:idx,:]
			df = df.drop( df.index[last_checkpoint:idx] )

		df_new.to_hdf( f"{self.overlapping_uni_pairs_dir}{new_df_name}.h5", key = "data", mode = "w" )
		del df_new
		
		return df


	def create_nonoverlapping_sets( self, upid ):
		"""
		Create a dataframe for the same.
		Sort the dataframe based on Uniprot residue start positions for both proteins.
			As a result of sorting a pair with the prot1 residues kept first,
				for some entries prot2 may not be sorted in order.
		Save all rows with overlapping protein seq (both) as a separate dataframe.
				Filename: Uniprot_ID1--Uniprot_ID2_copy_num.csv
		
		Input:
		----------
		upid --> a Uniprot ID pair. "{Uni_ID1}--{Uni_ID2}".

		Returns:
		----------
		nonoverlapping_entries --> (str) label for each non-overlapping entry.
							e.g. "{Uni_ID1}--{Uni_ID2}_{copy_num}"
		"""
		nonoverlapping_entries = []

		df = pd.read_hdf( f"{self.valid_binary_complexes_dir}{upid}.h5", dtype = self.dtype_dict )

		# Sort the df by Uniprot start residues.
		df = sort_by_residue_positions( df )

		# Just an integer to differentiate non-overlapping entries.
		copy_num = 0

		while( len( df ) != 0 ):
			last_checkpoint = 0
			df = df.reset_index( drop = True )

			# If only a single entry is present in the dataframe.
			if len( df ) == 1:
				df = self.save_overlapping_entries( df, f"{upid}_{copy_num}", last_checkpoint, None )
				nonoverlapping_entries.append( f"{upid}_{copy_num}" )
				copy_num += 1
			
			else:
				for idx in range( 1, len( df ) ):
					# Note: >1 residue overlap must be there.
					overlap1 = check_for_overlap( 
									df.loc[idx - 1, "Uniprot positions1"],
									df.loc[idx, "Uniprot positions1"]
													 )
					overlap2 = check_for_overlap( 
									df.loc[idx - 1, "Uniprot positions2"],
									df.loc[idx, "Uniprot positions2"]
													 )

					exceeds_maxlen1 = merged_seq_exceeds_maxlen( df.loc[idx - 1, "Uniprot positions1"],
										df.loc[idx, "Uniprot positions1"],
										self.max_len )
					exceeds_maxlen2 = merged_seq_exceeds_maxlen( df.loc[idx - 1, "Uniprot positions2"],
										df.loc[idx, "Uniprot positions2"],
										self.max_len )
					
					# For non-overlapping entries
					if any( [(not overlap1), (not overlap2), exceeds_maxlen1, exceeds_maxlen2] ):
						# Multiple overlapping entries before encountering a nonoverlapping entry.
						if ( idx - last_checkpoint ) > 1:
							df = self.save_overlapping_entries( df, f"{upid}_{copy_num}", last_checkpoint, idx - 1 )
							nonoverlapping_entries.append( f"{upid}_{copy_num}" )
							copy_num += 1
							break

						# Single entry with no overlap.
						else:
							df = self.save_overlapping_entries( df, f"{upid}_{copy_num}", last_checkpoint, None )
							nonoverlapping_entries.append( f"{upid}_{copy_num}" )
							copy_num += 1
							break

					# If all entries in a set are overlapping.
					elif idx == len( df ) - 1:
						df = self.save_overlapping_entries( df, f"{upid}_{copy_num}", last_checkpoint, idx+1 )
						nonoverlapping_entries.append( f"{upid}_{copy_num}" )
						copy_num += 1
						break

		del df
		return nonoverlapping_entries



	def module2( self ):
		"""
		Given a pandas dataframe containing all valid binary complexes 
			for all Uniprot ID pairs, create sets of overlapping 
			binary compexes (based on Uniprot position)
			and save each as a separate h5 file.
		
		Input:
		----------
		Does not take any input arguments.

		Returns:
		----------
		None
		"""
		non_overlaping_complexes, count_all_binary_complexes = 0, 0
		if not os.path.exists( self.overlapping_uni_pairs_dir ):
			os.makedirs( self.overlapping_uni_pairs_dir )
		
		# Load Uniprot ID pairs with valid binary complexes.
		with open( self.valid_binary_complexes_file, "r" ) as f:
			uni_id_pairs = f.readlines()[0].split( "," )

		selected_uni_pairs = []
		with Pool( self.cores ) as p:
			# Parallelize each Uniprot ID pair.
			for result in tqdm.tqdm( 
								p.imap_unordered( self.create_nonoverlapping_sets, 
												uni_id_pairs
												 ), 
							total = len( uni_id_pairs ) ):
				nonoverlapping_entries = result
				
				selected_uni_pairs.extend( nonoverlapping_entries )
		
		with open( self.overlapping_uni_pairs_file, "w" ) as w:
			w.writelines( ",".join( list( selected_uni_pairs ) ) )

		print( f"Unique Uniprot ID pairs = {len( selected_uni_pairs )}" )
		# print( f"Total binary complexes = {count_all_binary_complexes}" )

		# Count no. of binary complexes left.
		self.logger["counts"]["non_overlapping_set_creation"] = len( selected_uni_pairs )



###################################################################################################################
###################################################################################################################
###################################################################################################################
	def merge_contact_map( self, cmap_prev, cmap_curr, prot1_prev, prot1_curr, prot2_prev, prot2_curr,
							merged_prot1_pos, merged_prot2_pos, entry_id ):
		"""
		Merge the previous and current contact maps to obtain -
				merged_cmap --> simply 0 or 1 for contact or not.
		Create a zeros matrix of dim [len(merged_prot1_pos), len(merged_prot2_pos)].
		For both the previous and current entry:
			Obtain the start and end indices for prot1 residues in the merged_prot1_pos.
			Obtain the start and end indices for prot2 residues in the merged_prot2_pos.
		Merge the previous and current contact maps by using a logical OR operation on 
			the merged contact map slice for the corresponding start and end indices.
		A faster and efficient implementation than earlier.

		Input:
		----------
		cmap_prev --> (np.array) contact map for the previous entry.
		cmap_curr --> (np.array) contact map for the current entry.
		prot1_prev --> (np.array) prot1 Uniprot residue position for previous entry.
		prot1_curr --> (np.array) prot1 Uniprot residue position for current entry.
		prot2_prev --> (np.array) prot2 Uniprot residue position for previous entry.
		prot2_curr --> (np.array) prot2 Uniprot residue position for current entry.
		merged_prot1_pos --> (np.array) prot1 merged Uniprot residue position for previous entry.
		merged_prot2_pos --> (np.array) prot2 merged Uniprot residue position for previous entry.
		entry_id --> (str) Uniprot ID pair, same as input.

		Returns:
		----------
		merged_cmap --> (np.array) merged binary contact map.
		"""
		# Initialize a 0-matrix
		merged_cmap = np.zeros( ( len( merged_prot1_pos ), len( merged_prot2_pos ) ) )

		agg_cmap_curr = cmap_curr
		agg_cmap_prev = cmap_prev

		# Find the index of the start and end residues of prot1 and prot2
				# in the merged seq.
		# Merge the cmap at the appropriate start and end indices.
		# Current cmap.	
		start1 = np.where( merged_prot1_pos == prot1_curr[0] )[0][0]
		end1 = np.where( merged_prot1_pos == prot1_curr[-1] )[0][0]
		start2 = np.where( merged_prot2_pos == prot2_curr[0] )[0][0]
		end2 = np.where( merged_prot2_pos == prot2_curr[-1] )[0][0]
	
		# merged_cmap[start1:end1+1,start2:end2+1] = np.logical_or( merged_cmap[start1:end1+1,start2:end2+1], agg_cmap_curr )
		merged_cmap[start1:end1+1,start2:end2+1] = merged_cmap[start1:end1+1,start2:end2+1] + agg_cmap_curr

		# Previous cmap.
		start1 = np.where( merged_prot1_pos == prot1_prev[0] )[0][0]
		end1 = np.where( merged_prot1_pos == prot1_prev[-1] )[0][0]
		start2 = np.where( merged_prot2_pos == prot2_prev[0] )[0][0]
		end2 = np.where( merged_prot2_pos == prot2_prev[-1] )[0][0]

		# merged_cmap[start1:end1+1,start2:end2+1] = np.logical_or( merged_cmap[start1:end1+1,start2:end2+1], agg_cmap_prev )
		merged_cmap[start1:end1+1,start2:end2+1] = merged_cmap[start1:end1+1,start2:end2+1] + agg_cmap_prev

		return merged_cmap


	def merge_binary_complexes( self, entry_id, prev, curr ):
		"""
		Merge seq and contact map for the previous and current entry.
		In order to merge the seq for prot1 and prot2:
			Merge the Uniprot residue positions for the current and previous entry.
			Get the merged Uniprot seq using the merged Uniprot residue positions.
		Sanity check:
			Do not merge if:
			prot2 residue positions do not overlap (See self.create_nonoverlapping_sets()).
			Length of merged Uniprot seq and merged Uniprot 
				residue positions do not match.
			Length of merged seq > self.max_len.
		Merge the contact map for previous and current entry.

		Input:
		----------
		entry_id --> (str) Uniprot ID pair, same as input.
		prev --> list for the previous entry containing:
					sequences for prot1 and prot2.
					Uniprot residue positions for prot1 and prot2.
					binary contact map.
		curr --> list for the current entry containing:
					sequences for prot1 and prot2.
					Uniprot residue positions for prot1 and prot2.
					binary contact map.

		Returns:
		----------
		[merged_prot1_seq, merged_prot1_pos]
			merged_prot1_seq --> merged Uniprot seq for prot1.
			merged_prot1_pos --> merged Uniprot residue positions for prot1.
		[merged_prot2_seq, merged_prot2_pos]
			merged_prot2_seq --> merged Uniprot seq for prot2.
			merged_prot2_pos --> merged Uniprot residue positions for prot2.
		merged_cmap --> merged binary contact map.
		"""

		prot1_seq_prev, prot1_pos_prev, prot2_seq_prev, prot2_pos_prev, agg_cmap_prev = prev
		prot1_seq_curr, prot1_pos_curr, prot2_seq_curr, prot2_pos_curr, agg_cmap_curr = curr
		
		# Merge the prot1 chains from curr and prev entries.
		prot1_uniprot_id, prot2_uniprot_id = entry_id.split( "--" )
		prot2_uniprot_id = prot2_uniprot_id.split( "_" )[0]
		
		# merged_prot1_pos = np.sort( np.unique( np.concatenate( ( prot1_pos_prev, prot1_pos_curr ) ) ) )
		merged_prot1_pos = merge_residue_positions( prot1_pos_prev, prot1_pos_curr )
		merged_prot1_seq = self.accio_uni_seq( prot1_uniprot_id, merged_prot1_pos )
		
		overlap = check_for_overlap( prot2_pos_prev, prot2_pos_curr, 
									ignore_boundary = True )
		
		# If the protein2 Uniprot positions are not overlapping, do not merge.
		if overlap:
			merged_prot2_pos = merge_residue_positions( prot2_pos_prev, prot2_pos_curr )
			merged_prot2_seq = self.accio_uni_seq( prot2_uniprot_id, merged_prot2_pos )
		
			# Remove if length of Uniprot seq and residue positions do not match.
			if len( merged_prot1_seq ) != len( merged_prot1_pos ) or len( merged_prot2_seq ) != len( merged_prot2_pos ):
				print( "--> ", len( merged_prot1_pos ), "\t", len( merged_prot2_pos ), "\t", len( merged_prot1_seq ), "\t", len( merged_prot2_seq ) )
				return 0, 0, 0
			
			# Don't merge if prot1/2 merged lengths >= max_len.
			elif len( merged_prot1_seq ) > self.max_len or len( merged_prot2_seq ) > self.max_len:
				return None, None, None
			
			else:
				# Merge the contact maps for heterogenous entries.
				merged_cmap = self.merge_contact_map( agg_cmap_prev, agg_cmap_curr, 
											prot1_pos_prev, prot1_pos_curr,
											 prot2_pos_prev, prot2_pos_curr,
											 merged_prot1_pos, merged_prot2_pos, entry_id )
				return [merged_prot1_seq, merged_prot1_pos], [merged_prot2_seq, merged_prot2_pos], merged_cmap
		else:
			return [], [], []


	def create_merged_binary_complexes( self, entry_id ):
		"""
		Parse through all the binary complexes for a given Uniprot ID pair 
				stored in the respective .h5  files.
				identified by "Uniprot_ID1--Uniprot_ID2_{copy_num}".
				This forms the entry_id.
		Load the relevant data PDB ID, Chain IDs, Uniprot and PDB positions.
		Merge entries which have overlapping Uniprot residue positions.
			Starting from the first entry we merge all subsequent entry until 
					none can be merged and save the same on disk.

		Sanity checks:
				Prot2 residue positions should overlap.
				Merged length should not exceed max_length bound (checking for prot2 only).
				Length of merged seq and residue positions should match.
				Length of seq for proteins and the respective cmap dimension should match.

		Data saved for a selected entry includes:
				Merged sequences and their lengths
				Binary contact map
				Uniprot boundaries
				Contact count
				No. of binary complexes merged
		"""
		#### Creating contact maps and merging heterogeneity. ####
		logs_dict = {i:[] for i in [
									"missing_chain_in_model", "uni_seq_pos_mismatch", 
									"mismatch_cmap_seq_dim", 
									"mismatch_prot1/prot2_length", "no_overlap_prot2_uni_pos", 
									"merged_len_exceed", "hetero", "cargo"]}

		if not os.path.exists( f"{self.merged_binary_complexes_logs_dir}{entry_id}.json" ):
			# Load the sorted entries for the given entry_id.
			df_sorted = pd.read_hdf( f"{self.overlapping_uni_pairs_dir}{entry_id}.h5", dtype = self.dtype_dict )

			# Load all unique PDBs.
			unique_pdbs = pd.unique( df_sorted["PDB ID"] )
			pdbs_dict = self.create_PDBs_dict( unique_pdbs )

			# Keep track of all the conformers across all overlapping entries.
			total_conformers = 0
			prev = [] # List to store previous entry for merging.
			accepted_entries = []

			all_indexes = df_sorted.index
			for entry_idx in all_indexes:
				pdb = df_sorted["PDB ID"][entry_idx]
				
				uni1_id, uni2_id = entry_id.split( "--" )
				uni2_id, _ = uni2_id.split( "_" )

				# Get Auth Asym ID for PDB files.
				chain1_id = df_sorted["Auth Asym ID1"][entry_idx]
				chain2_id =  df_sorted["Auth Asym ID2"][entry_idx]			

				# An identifier for each binary complex belonging to a Uni ID pair.
				# {UniID1}--{UniID2}_{copy_num}_{index}_{PDB ID}:{Chain1}:{Chain2}
				entry_key = f"{entry_id}_{entry_idx}_{pdb}:{chain1_id}:{chain2_id}"

				# Get the PDB residue positions.
				uni1_pos = df_sorted["Uniprot positions1"][entry_idx]
				uni2_pos = df_sorted["Uniprot positions2"][entry_idx]

				# Get the Uniprot seq for prot1 and prot2.
				prot1_seq = self.accio_uni_seq( uni1_id, uni1_pos )
				prot2_seq = self.accio_uni_seq( uni2_id, uni2_pos )
				
				# Get the PDB residue positions.
				pdb1_pos = df_sorted["PDB positions1"][entry_idx]
				pdb2_pos = df_sorted["PDB positions2"][entry_idx]

				#### Get the aggregate contact_map.
				####------------------------------------------------------
				chain_missing, excess_res_coords, num_conformers, agg_cmap  = self.create_contact_map(
																						pdbs_dict[pdb], 
																						pdb1_pos, pdb2_pos,
																						chain1_id, chain2_id,
																						all_models = self.all_models )


				# Clearing memory.
				df_sorted = df_sorted.drop( entry_idx )
				
				#### Merge heterogenous entries.
				####------------------------------------------------------
				if prev == []:
					prev = [prot1_seq, uni1_pos, prot2_seq, uni2_pos, agg_cmap]
					merged_prot1_seq, merged_prot1_pos = prot1_seq, uni1_pos
					merged_prot2_seq, merged_prot2_pos = prot2_seq, uni2_pos
					merged_cmap = agg_cmap
		
					# Ignore is there is a mismatch in cmap dim and merged seq lengths.
					if len( merged_prot1_seq ) != merged_cmap.shape[0] or len( merged_prot2_seq ) != merged_cmap.shape[1]:
						logs_dict["mismatch_cmap_seq_dim"].append( entry_key )
						continue
					else:
						total_conformers += num_conformers
						accepted_entries.append( entry_key )

				else:
					curr = [prot1_seq, uni1_pos, prot2_seq, uni2_pos, agg_cmap]
					# Fetch the contact map for the hetergenous entry with same IDR and R.
					prot1_seq_pos, prot2_seq_pos, merged_cmap_ = self.merge_binary_complexes(
																							entry_id, 
																							prev, 
																							curr )

					# If not merged due to exceeding max len, ignore the entry -- Sanity check.
					# 	Overlaping seq with different cmaps might be confusing for the model.
					# For exceeding max_len None is returned for all 3 returned values.
					if prot1_seq_pos == None:
						logs_dict["merged_len_exceed"].append( entry_key )
						continue

					# Ignore an entry if the length of merged uni seq and merged uni pos do not match. -- Sanity check
					# For mismatch in merged uni seq and pos 0 is returned for all 3 returned values.
					elif prot1_seq_pos == 0:
						logs_dict["uni_seq_pos_mismatch"].append( entry_key )
						continue

					# Do not merge if the prev and curr protein2 Uniprot positions do not overlap.
					elif len( prot1_seq_pos ) == 0:
						logs_dict["no_overlap_prot2_uni_pos"].append( entry_key )
						continue

					# Ignore if there is a mismatch in cmap dim and merged seq lengths -- Sanity check.
					elif len( prot1_seq_pos[0] ) != merged_cmap_.shape[0] or len( prot2_seq_pos[0] ) != merged_cmap_.shape[1]:
						logs_dict["mismatch_cmap_seq_dim"].append( entry_key )
						continue

					else:
						logs_dict["hetero"].append( entry_key )
						accepted_entries.append( entry_key )
						
						merged_prot1_seq, merged_prot1_pos = prot1_seq_pos
						merged_prot2_seq, merged_prot2_pos = prot2_seq_pos
						merged_cmap = merged_cmap_

						prev = [merged_prot1_seq, merged_prot1_pos, merged_prot2_seq, merged_prot2_pos, merged_cmap]
						total_conformers += num_conformers

				# If self.no_hetero is True, ignoring heterogeneity i.e.
				# 	Considering only 1 conformer per Uniprot ID pair.
				# 	Break out as soon as the first conformer is selected.
				if self.no_hetero and len( prev ) != 0:
					break

			if accepted_entries != []:
				logs_dict["cargo"].append( entry_id )

				uni1_boundary = f"{merged_prot1_pos[0]}-{merged_prot1_pos[-1]}"
				uni2_boundary = f"{merged_prot2_pos[0]}-{merged_prot2_pos[-1]}"
				hf = h5py.File( f"{self.merged_binary_complexes_dir}{entry_id}.h5", "w" )
				hf.create_dataset( "prot1_seq", data = merged_prot1_seq )
				hf.create_dataset( "prot1_uni_boundary", data = uni1_boundary )
				hf.create_dataset( "prot1_length", data = len( merged_prot1_seq ) )
				hf.create_dataset( "prot2_seq", data = merged_prot2_seq )
				hf.create_dataset( "prot2_uni_boundary", data = uni2_boundary )
				hf.create_dataset( "prot2_length", data = len( merged_prot2_seq ) )
				hf.create_dataset( "conformers", data = total_conformers )
				hf.create_dataset( "merged_entries", data = accepted_entries )
				hf.create_dataset( "merged_entries_count", data = len( accepted_entries ) )
				hf.create_dataset( "summed_cmap", data = merged_cmap )
				
				merged_cmap = np.where( merged_cmap > 0, 1, 0 )
				hf.create_dataset( "contacts_count", data = np.count_nonzero( merged_cmap ) )
				hf.create_dataset( "binary_cmap", data = merged_cmap )
				hf.close()
				# Clear from memory.
				del merged_cmap

			with open( f"{self.merged_binary_complexes_logs_dir}{entry_id}.json", "w" ) as w:
				json.dump( logs_dict, w )

		# Load if the upid has already been processed.
		else:
			with open( f"{self.merged_binary_complexes_logs_dir}{entry_id}.json", "r" ) as f:
				logs_dict = json.load( f )

		return [logs_dict]



	def module3( self ):
		"""
		For all Uniprot ID pairs, do:
			Create contact maps.
			Merge sequences and contact maps for all binary complexes.
		See self.create_merged_binary_complexes() for info collected.

		**Note
		Loading large CIF files in parallel causes significantly huge memory consumption.
			e.g. 8glv, 8oj8, 8j07, 8qo9, etc.
		Most of the Uniprot ID pairs containing such large CIF files are hoarded towards the end.
		Thus we process them in smaller batches that can be accomodated in memory.
		Loading and processing them serially will take a large amount of time.

		Input:
		----------
		Does not take any input arguments.

		Returns:
		----------
		None
		"""
		if not os.path.exists( self.merged_binary_complexes_dir ):
			os.makedirs( self.merged_binary_complexes_dir )
		if not os.path.exists( self.merged_binary_complexes_logs_dir ):
			os.makedirs( self.merged_binary_complexes_logs_dir )
		
		# Load overlapping Uniprot ID pairs.
		with open( self.overlapping_uni_pairs_file, "r" ) as f:
			uni_id_pairs = f.readlines()[0].split( "," )

		selected_merged_pairs = []
		
		# Processing batches in parallel.
		# Creating a major first batch and subsequent small batches 
		# 			to efficiently handle the large CIF containing entries.
		batches = np.arange( 0, len( uni_id_pairs ), self.m3_batch_size )
		for start in batches:
			end = start + self.m3_batch_size
			batch = uni_id_pairs[start:end]

			with Pool( self.cores ) as p:
				for result in tqdm.tqdm( p.imap_unordered( self.create_merged_binary_complexes, 
															batch ), total = len( batch ) ):

					logs_dict = result[0]

					for key in logs_dict.keys():
						if logs_dict[key] != []:
							if key == "cargo":
								selected_merged_pairs.extend( logs_dict[key] )
							else:
								self.logger[key][0].extend( logs_dict[key] )
								self.logger[key][1] += len( logs_dict[key] )

			print( f"Completed for batch {start}-{end}..." )
		
		with open( self.merged_binary_complexes_file, "w" ) as w:
			w.writelines( ",".join( selected_merged_pairs ) )

		# Count no. of merged binary complexes.
		self.logger["counts"]["merged_binary_complexes"] = len( selected_merged_pairs )



###################################################################################################################
###################################################################################################################
###################################################################################################################
	def save_logs( self ):
		"""
		Save the logs.

		Input:
		----------
		Does not take any input arguments.

		Returns:
		----------
		None
		"""
		proc = subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE, )
		system = proc.communicate()[0]
		proc = subprocess.Popen( "date", shell = True, stdout = subprocess.PIPE )
		sys_date = proc.communicate()[0]

		self.tom = time.time()
		self.logger["time_taken"]["total"] = self.tom-self.tim
		print( "Total time taken = ", self.logger["time_taken"]["total"] )
		# Create the log file.
		with open( f"Logs_{self.version}.txt", "w" ) as w:
			w.writelines( "<------------------------Logs for Disobind Dataset------------------------>\n\n" )
			w.writelines( f"System = {system.strip()} \n" )
			w.writelines( f"Date = {sys_date} \n" )
			w.writelines( "\n---------------------------------------------------------\n" )
			w.writelines( "------------------Settings used\n" )
			w.writelines( f"No. of cores = {self.cores}\n" )
			# w.writelines( f"Batch size = {self.batch_size}\n" )
			w.writelines( f"Min seq length = {self.min_len}\n" )
			w.writelines( f"Max seq length = {self.max_len}\n" )
			w.writelines( f"Contact threshold = {self.contact_threshold}\n" )
			w.writelines( f"Select all models = {self.all_models}\n" )
			w.writelines( f"Not considering heterogeneity = {self.no_hetero}\n" )
			w.writelines( "\n---------------------------------------------------------\n" )
			w.writelines( "------------------Time taken\n" )
			w.writelines( "Creating valid binary complexes: %s minutes\n"%( self.logger["time_taken"]["valid_binary_complexes"]/60 ) )
			w.writelines( "Creating overlapping Uniprot ID pairs: %s minutes\n"%( self.logger["time_taken"]["non_overlapping_set_creation"]/60 ) )
			w.writelines( "Creating merged binary complexes: %s hours\n"%( self.logger["time_taken"]["merged_binary_complexes"]/3600 ) )
			w.writelines( "Total time taken: %s hours\n"%( self.logger["time_taken"]["total"]/3600 ) )
			w.writelines( "\n---------------------------------------------------------\n" )
			w.writelines( "------------------Counts\n" )
			w.writelines( "Total binary complexes: %s\n"%self.logger["counts"]["total_binary_complexes"] )
			w.writelines( "Valid Uniprot ID pairs: %s\n"%self.logger["counts"]["valid_uniprot_ID_pairs"] )
			w.writelines( "Valid binary complexes: %s\n"%self.logger["counts"]["valid_binary_complexes"] )
			w.writelines( "Total Non-overlapping Uniprot ID pairs: %s\n"%self.logger["counts"]["non_overlapping_set_creation"] )
			w.writelines( "Total Merged binary complexes: %s\n"%self.logger["counts"]["merged_binary_complexes"] )
			w.writelines( "\n---------------------------------------------------------\n" )

			for key in self.logger.keys():
				if key not in ["time_taken", "counts"]:
					w.writelines( "-->" + key + "\n" )
					w.writelines( "\nCount = %s\n"%self.logger[key][1] )
					w.writelines( "========================================================\n" )

		# Save logger state.
		with open( self.logger_file, "w" ) as w:
			json.dump( self.logger, w )



if __name__ == "__main__":
	parser = argparse.ArgumentParser( description="Create Merged binary complexes." )
	parser.add_argument( '--version', '-v', dest = "v", help = "Version of the dataset", 
						type = str, required = True )
	parser.add_argument( '--max_cores', '-c', dest="c", help = "No. of cores to be used.", 
						type = int, required = False, default = 10 )
	
	version = parser.parse_args().v
	cores = parser.parse_args().c
	
	Dataset( version, cores ).forward()

	print( "May the Force b with u..." )

"""
【中文解析-可演化改动建议】
1) 将硬编码路径迁移到统一配置文件（YAML/ENV）并加入路径合法性校验。
2) 为每个中间文件增加 schema 校验（字段名/类型/范围），降低跨脚本耦合导致的隐式错误。
3) 增加小样本 smoke test（例如 3~5 条记录）和数据快照测试，保障重构时行为一致。
4) 若面向预训练，建议输出 token 级别元信息（来源数据库、结构置信度、对齐覆盖率）便于构建多任务标签。
"""
