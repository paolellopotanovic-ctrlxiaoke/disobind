"""
【中文解析-模块总览】
- 中心功能：prep_idppi_input2.py 属于 Disobind 数据流水线脚本，用于数据抓取/清洗/映射/导出中的一个环节。
- 逻辑位置：该文件在“原始数据库 -> 统一样本定义 -> 训练/评估输入文件”链条中承担局部处理任务。
- 输入输出流水线：通常输入为数据库文件（csv/tsv/xlsx/json）、结构文件（pdb/cif）与序列标识（UniProt/PDB），输出为中间表、映射字典、模型可读输入（csv/h5/json/npy/fasta）。
- 可选项：多数脚本支持通过构造参数或命令行参数控制版本号、并行核数、路径和过滤阈值。
- 可视化能力：数据脚本本身通常不直接出图；如有图像分析一般在 analysis/ 目录统一完成。
- 数据格式与实验标签：核心监督标签通常是残基-残基接触图（contact map）或界面位点（interface residues）；本目录负责把外部异构数据规范化成论文实验使用的统一格式。
- 数据来源与论文逻辑：数据来源包含 DIBS/MFIB/DisProt/IDEAL/MobiDB/PDB/SIFTS 等；处理逻辑服务于“按二元复合体构建样本并对齐序列-结构关系”。
- 预训练扩展建议：若做进一步预训练，建议先扩展原始来源覆盖面、统一长度分桶策略、保留更多中间质控字段（置信度/覆盖率/冲突标记），再生成更大规模弱监督样本。
"""

"""
Create Disobind input files for the IDPPI dataset.
DOI: https://doi.org/10.1038/s41598-018-28815-x
"""
from typing import List, Tuple, Dict
import os, json
import numpy as np
import pandas as pd
from multiprocessing import Pool
import tqdm

from dataset.from_APIs_with_love import get_uniprot_seq


# 【中文解析-关键函数/类入口】以下对象是本文件的主执行入口，承担数据转换主链路。
class IdppiInput():
	"""
	Evaluate Disobind on the IDPPI dataset using interface_1 model.
	"""
	def __init__( self ):
		self.base_dir = "../database/"
		self.idppi_output_dir = os.path.join( self.base_dir, "./idppi/" )

		self.idppi_file = os.path.join( self.base_dir, "input_files/41598_2018_28815_MOESM2_ESM.xlsx" )
		self.diso_uni_seq_file = os.path.join( self.base_dir, "v_19/Uniprot_seq.json" )

		self.cores = 100
		self.max_seq_len = 10000
		# If True, ignores protein pairs which are present in Disobind dataset.
		self.remove_diso_seq = True
		
		self.uniprot_seq_dict = {}
		self.logger = {}

		# .json file containing Uniprot sequences for IDPPI dataset.
		self.uniprot_seq_file = os.path.join( self.idppi_output_dir, "Uniprot_seq_idppi.json" )
		# .csv file containing IDPPI entry_ids for Disobind.
		self.diso_input_file = os.path.join( self.idppi_output_dir, "IDPPI_input_diso.csv" )
		self.idppi_target_file = os.path.join( self.idppi_output_dir, "IDPPI_target.json" )
		self.logs_file = os.path.join( self.idppi_output_dir, "Logs.txt" )
	# 【中文解析-重点逻辑】forward 通常串联“读取输入 -> 清洗/映射 -> 导出中间/最终文件”，是该脚本最关键的数据流水线节点。
	def forward( self ):
		"""
		Parse the IDPPI .xslx file.
		Download the unique UniProt seq.
		Remove protein pairs for which UniProt seq was not obtained.
		"""
		self.create_dir()
		idppi_pairs, unique_uni_ids = self.parse_idppi_file()
		self.logger["idppi_pairs"] = len( idppi_pairs )
		self.logger["unique_uni_ids"] = len( unique_uni_ids )

		print( f"Total protein pairs obtained = {len( idppi_pairs )}" )
		print( f"Unique UniProt IDs obtained = {len( unique_uni_ids )}" )

		self.get_uniprot_seq_dict( unique_uni_ids )
		idppi_test_dict = self.filter_idppi_pairs( idppi_pairs )

		self.logger["selected_idppi_pairs"] = len( idppi_test_dict )
		print( f"Selected IDPPI protein pairs = {len( idppi_test_dict )}" )

		self.create_input_for_disobind( idppi_test_dict )
		self.write_logs()


	def create_dir( self ):
		"""
		Create the required directories.
		"""
		os.makedirs( self.idppi_output_dir, exist_ok = True )


	def parse_idppi_file( self ) -> Tuple[Dict, List]:
		"""
		Obtain all protein pairs and the corresponding labels from the 5 IDPPI Test sets.
		Obtain all entries in the Test sets (1-5)
			in sheets 17-21.
		"""
		with open( self.diso_uni_seq_file, "r" ) as f:
			diso_uni_seq_dict = json.load( f )

		idppi_pairs = {}
		unique_uni_ids = set()
		for sheet_name in ["Table S17-TestSet1", "Table S18-TestSet2 ",
							"Table S19-TestSet3 ", "Table S20-TestSet4 ",
							"Table S21-TestSet5"]:
			df = pd.read_excel( self.idppi_file, sheet_name = sheet_name, header = None )
			print( sheet_name, "  ", df.shape[0] )
			for row in df[0].str.split( " " ):
				pair_id = f"{row[1]}--{row[2]}"
				if self.remove_diso_seq:
					# Remove protein  pairs if they are present in Disobind dataset.
					if row[1] in diso_uni_seq_dict or row[2] in diso_uni_seq_dict:
						continue
				idppi_pairs[pair_id] = row[0]
				unique_uni_ids.update( row[1:] )
		
		return idppi_pairs, sorted( unique_uni_ids )


	def download_uniprot_seq( self, uni_id: str ) -> Tuple[str, str, bool]:
		"""
		Get the UniProt seq given the UniProt ID.
		"""
		uni_seq = get_uniprot_seq( uni_id, max_trials = 5, wait_time = 5, return_id = False )
		if len( uni_seq ) == 0:
			success = False
		else:
			success = True
		return uni_id, uni_seq, success


	def get_uniprot_seq_dict( self, unique_uni_ids: List ):
		"""
		Download all UniProt IDs.
		"""
		if os.path.exists( self.uniprot_seq_file ):
			print( "Loading pre-downloaded UniProt seq..." )
			with open( self.uniprot_seq_file, "r" ) as f:
				self.uniprot_seq_dict = json.load( f )
		else:
			with Pool( self.cores ) as p:
				for result in tqdm.tqdm( p.imap_unordered( self.download_uniprot_seq, unique_uni_ids ),
															total = len( unique_uni_ids ) ):

					uni_id, uni_seq, success = result
					if success:
						self.uniprot_seq_dict[uni_id] = uni_seq

			with open( self.uniprot_seq_file, "w" ) as w:
				json.dump( self.uniprot_seq_dict, w )

		self.logger["uni_seq_dwnld"] = len( self.uniprot_seq_dict )
		print( f"UniProt sequences obtained = {len( self.uniprot_seq_dict )}" )


	def filter_idppi_pairs( self, idppi_pairs: Dict ) -> Dict:
		"""
		Ignore protein pairs for which:
			UniProt seq could not be downloaded.
			UniProt seq len > max_seq_len.
		"""
		idppi_test_dict = {}
		for pair_id in idppi_pairs:
			uni_id1, uni_id2 = pair_id.split( "--" )
			if uni_id1 not in self.uniprot_seq_dict or uni_id2 not in self.uniprot_seq_dict:
				continue
			elif ( 
					( len( self.uniprot_seq_dict[uni_id1] ) > self.max_seq_len ) or 
					( len( self.uniprot_seq_dict[uni_id1] ) > self.max_seq_len )
					):
				continue
			else:
				idppi_test_dict[pair_id] = idppi_pairs[pair_id]
		return idppi_test_dict


	def create_input_for_disobind( self, idppi_test_dict: Dict ):
		"""
		For all selected protein pairs from IDPPI test set,
			create entry_id for input to Disobind.
		"""
		disobind_input_pairs = []
		for pair_id in idppi_test_dict:
			uni_id1, uni_id2 = pair_id.split( "--" )

			seq_len1 = len( self.uniprot_seq_dict[uni_id1] )
			seq_len2 = len( self.uniprot_seq_dict[uni_id2] )

			copy_num = 0
			entry_id = f"{uni_id1}:{1}:{seq_len1}--{uni_id2}:{1}:{seq_len2}_{copy_num}"

			disobind_input_pairs.append( entry_id )

		print( len( disobind_input_pairs ) )
		self.logger["total_idpi_entry_ids"] = len( disobind_input_pairs )
		with open( self.diso_input_file, "w" ) as w:
			w.writelines( ",".join( disobind_input_pairs ) )
		with open( self.idppi_target_file, "w" ) as w:
			json.dump( idppi_test_dict, w )

		target_labels = list( idppi_test_dict.values() )
		total = len( target_labels )
		pos_pairs = np.count_nonzero( np.array( target_labels ).astype( int ) )
		self.logger["interacting_pairs_count"] = pos_pairs
		self.logger["non_interacting_pairs_count"] = total-pos_pairs


	def write_logs( self ):
		"""
		Write logs to a .txt file.
		"""
		with open( self.logs_file, "w" ) as w:
			w.writelines( "------------------- IDPPI Logs -------------------\n" )
			w.writelines( "Configs -----\n" )
			w.writelines( f"Cores: {self.cores}\n" )
			w.writelines( f"Max sequence len: {self.max_seq_len}\n" )
			w.writelines( f"Redundancy reduce with v_21 UniProt seq: {self.remove_diso_seq}\n" )
			w.writelines( "\nStats -----\n" )
			w.writelines( f"Total IDPPI protein pairs = {self.logger['idppi_pairs']}\n" )
			w.writelines( f"Unique IDPI UniProt IDs obtained = {self.logger['unique_uni_ids']}\n" )
			w.writelines( f"Total UniProt seq obtained = {self.logger['uni_seq_dwnld']}\n" )
			w.writelines( f"Selected IDPPI pairs = {self.logger['selected_idppi_pairs']}\n" )
			w.writelines( f"Total IDPPI entry_ids obtained = {self.logger['total_idpi_entry_ids']}\n" )
			w.writelines( f"\tTotal Interacting pairs = {self.logger['interacting_pairs_count']}\n" )
			w.writelines( f"\tTotal non-Interacting pairs = {self.logger['non_interacting_pairs_count']}\n" )


if __name__ == "__main__":
	IdppiInput().forward()

"""
【中文解析-可演化改动建议】
1) 将硬编码路径迁移到统一配置文件（YAML/ENV）并加入路径合法性校验。
2) 为每个中间文件增加 schema 校验（字段名/类型/范围），降低跨脚本耦合导致的隐式错误。
3) 增加小样本 smoke test（例如 3~5 条记录）和数据快照测试，保障重构时行为一致。
4) 若面向预训练，建议输出 token 级别元信息（来源数据库、结构置信度、对齐覆盖率）便于构建多任务标签。
"""
