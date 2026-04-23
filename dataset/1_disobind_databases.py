"""
【中文解析-模块总览】
- 中心功能：1_disobind_databases.py 属于 Disobind 数据流水线脚本，用于数据抓取/清洗/映射/导出中的一个环节。
- 逻辑位置：该文件在“原始数据库 -> 统一样本定义 -> 训练/评估输入文件”链条中承担局部处理任务。
- 输入输出流水线：通常输入为数据库文件（csv/tsv/xlsx/json）、结构文件（pdb/cif）与序列标识（UniProt/PDB），输出为中间表、映射字典、模型可读输入（csv/h5/json/npy/fasta）。
- 可选项：多数脚本支持通过构造参数或命令行参数控制版本号、并行核数、路径和过滤阈值。
- 可视化能力：数据脚本本身通常不直接出图；如有图像分析一般在 analysis/ 目录统一完成。
- 数据格式与实验标签：核心监督标签通常是残基-残基接触图（contact map）或界面位点（interface residues）；本目录负责把外部异构数据规范化成论文实验使用的统一格式。
- 数据来源与论文逻辑：数据来源包含 DIBS/MFIB/DisProt/IDEAL/MobiDB/PDB/SIFTS 等；处理逻辑服务于“按二元复合体构建样本并对齐序列-结构关系”。
- 预训练扩展建议：若做进一步预训练，建议先扩展原始来源覆盖面、统一长度分桶策略、保留更多中间质控字段（置信度/覆盖率/冲突标记），再生成更大规模弱监督样本。
"""

######### Parse sequence and structure databases to get  #########
######### lists of PDB IDs containing IDRs in complex      ##########
######### ------>"May the Force serve u well..." <------##########
##################################################################

############# One above all #############
##-------------------------------------##

import numpy as np
import pandas as pd
import math
import random
from datetime import date
import argparse
import tqdm

import json
from functools import partial
from multiprocessing import Pool, get_context
import tqdm

from from_APIs_with_love import *
import xml.etree.ElementTree as ET

import warnings
warnings.filterwarnings( "ignore" )
random.seed( 11 )

##########################################################################
#------------------------------------------------------------------------#
# 【中文解析-关键函数/类入口】以下对象是本文件的主执行入口，承担数据转换主链路。
class TheIlluminati():
	"""
	Get the list of PDBs and associated information from 
		databases on structures of IDR complexes.
	DIBS, MFIB, FuzDB, PDBtot, PDBcdr are considered here.  
	"""

	def __init__( self, dir_name ):
		"""
		Constructor. 
		The raw files (self.dibs, self.mfib, self.fuzzdb) etc 
		are downloaded as is from the web servers or papers of the databases. 
		The output_file containts a list of PDBs and associated information parsed from the raw files.
		"""

		self.base_path = dir_name # "./input_files/"
		self.dibs    = "./raw/DIBS_complete_17Apr24.txt"         # DIBS txt file  path.
		self.mfib    = "./raw/MFIB_complete_17Apr24.txt"         # MFIB txt file path.
		self.fuzzdb  = "./raw/browse_fuzdb.tsv"                  # Fuzzdb tsv file path.
		self.pdb_tot = "./raw/FP_pdbtot_modified.xlsx"           # PDB TOT xlsx file path.
		self.pdb_cdr = "./raw/FP_pdbcdr_modified.xlsx"           # PDB CDR xlsx file path.
		
		self.dibs_mfib_csv = f"{self.base_path}DIBS_MFIB.csv"
		self.pdb_tot_cdr_csv = f"{self.base_path}PDB_tot_cdr.csv"
		self.output_file = f"Merged_DIBS_MFIB_Fdb_PDBtot-cdr.csv"


	def convert_txt_to_dict( self, txt ):
		"""
		Convert the database txt files to csv with relevant info.
		Input:
		----------
		txt --> handle for the txt file. 

		Returns:
		----------
		dict --> all the database info is returned as a dictionary
		""" 
	
		# Create a dict containing lists for all entry headers
		# entry_dict = {i:[] for i in self.concat_dict.keys()}
		entry_dict = {i:[] for i in ["Accession", "PDB ID", "Asym ID1", "Asym ID2", \
		"Auth Asym ID1", "Auth Asym ID2", "Uniprot accession1", "Uniprot accession2", "Uniprot boundary1", "Uniprot boundary2"] }

		# Add each entry into the respective header
		for i in range( len( txt ) ):
			if "[Entry]" in txt[i]:
				
				# Loop over for each entry identified by [Entry]
				for j in range( i+1, len( txt ) ):
					
					# A newline charcter (\n) separates 2 entries
					# so break the loop if a \n character is encountered.
					if "\n" == txt[j]:
						break

					elif txt[j].startswith( "[Accession]" ):
						# PDB ID for the complex
						entry_dict["Accession"].append( txt[j].strip().split( "=" )[1] )
					
					elif txt[j].startswith( "[PDB ID]" ):
						# PDB ID for the complex
						entry_dict["PDB ID"].append( txt[j].strip().split( "=" )[1] )
					
					elif txt[j].startswith( "[PDB chain IDs]" ):
						# PDB chains for the disordered and ordered component(s)
						# chain ID1 for disordered component
						# chain ID2 for ordered component(s)
						# Some entries have >1 ordered component
						ids = txt[j].strip().split( "=" )[1]
						
						# Chain IDs in DIBS are separated by a ":" for IDR and R.
						# ":" not present in chain ids for MFIB enties.
						if ":" in ids:
							chain_id1 = ids[0]
							entry_dict["Auth Asym ID1"].append( ",".join( chain_id1 ) )

							chain_id2 = ids.split( ":" )[1]
							entry_dict["Auth Asym ID2"].append( ",".join( chain_id2 ) )
						else:
							# For MFIB entries, consider all chains as IDR.
							chain_id1 = ids
							entry_dict["Auth Asym ID1"].append( ",".join( chain_id1 ) )
							chain_id2 = ""

						uni_idr, uni_r, uni_bound1, uni_bound2 = [], [], [], []
						for k in range( j+1, len( txt ) ):
							# Loop over all the headers in the entry
							if txt[k] == "\n":
								# Break if end of entry reached
								break
							
							# For MFIB entries, chain_id2 will be empty.
							elif chain_id2 == "":
								for chain in chain_id1:
									if f"[Chain {chain} UniProt accession]" in txt[k]:
										tmp = txt[k].strip().split( "=" )[1].split( "," )
										[uni_idr.append( x )  for x in tmp]

									elif f"[Chain {chain} UniProt boundaries]" in txt[k]:
										tmp = txt[k].strip().split( "=" )[1].split( "," )
										[uni_bound1.append( x )  for x in tmp]

							elif f"[Chain {chain_id1} UniProt accession]" in txt[k]:
								# Get the Uniprot ID for the 1st chain
								uni_idr.append( txt[k].strip().split( "=" )[1] )

							elif f"[Chain {chain_id1} UniProt boundaries]" in txt[k]:
								# Get the Uniprot boundaries for the 1st chain
								uni_bound1.append( txt[k].strip().split( "=" )[1] )

							else:
								for chains in chain_id2:
									# Appending the info for each chain in a list.
									# Order of entries in the list represents the order in which the
									# R chains appear in the DIBS db txt file.
									# eg: chains: BA; so uniprot ID: P09O## (for B), P08O## (for A)

									if f"[Chain {chains} UniProt accession]" in txt[k]:
										# Get the Uniprot ID for the 2nd chain
										tmp = txt[k].strip().split( "=" )[1].split( "," )
										[uni_r.append( x )  for x in tmp]

									if f"[Chain {chains} UniProt boundaries]" in txt[k]:
										# Get the Uniprot ID for the 2nd chain
										tmp = txt[k].strip().split( "=" )[1].split( "," )
										[uni_bound2.append( x )  for x in tmp]
						
						entry_dict["Uniprot accession1"].append( ",".join( uni_idr ) )
						entry_dict["Uniprot boundary1"].append( ",".join( uni_bound1 ) )
						# Uniprot accession2 will be empty for MFIB.
						if uni_r == []:
							entry_dict["Uniprot accession2"].append( uni_r )
							entry_dict["Uniprot boundary2"].append( uni_bound2 )
						else:
							entry_dict["Uniprot accession2"].append( ",".join( uni_r ) )
							entry_dict["Uniprot boundary2"].append( ",".join( uni_bound2 ) )

			else:
				continue

		return entry_dict


	def dibs_mfib( self, dibs = True, mfib = True ):
		"""
		DIBS contains complexes formed by disordered proteins with their ordered binding partner.
		MFIB contains complexes formed by disordered proteins only.
		Extract PDB IDs and associated info from DIBS and MFIB.
		"""

		# Dict for the combined DIBS+MFIB database.
		concat_dict = {i:[] for i in ["PDB ID", "Asym ID1", "Asym ID2", \
		"Auth Asym ID1", "Auth Asym ID2", "Uniprot accession1", "Uniprot accession2", "Uniprot boundary1"] }

		if dibs and mfib:
			db = [self.dibs, self.mfib]
		elif dibs:
			db = [self.dibs]
		else:
			db = [self.mfib]

		# For both the databases - DIBS and MFIB.		
		for txt in db:
			with open( txt, "r" ) as f:
				txt = f.readlines()

			tmp_dict = self.convert_txt_to_dict( txt )

			for key in concat_dict.keys():
				# If any key is empty, fill it with "None".

				if tmp_dict[key] == [] or tmp_dict[key][0] == []:
					tmp_dict[key] = []
					[tmp_dict[key].append( "None" ) for i in range( len( tmp_dict["PDB ID"] ) )]
				
				# Concatenate the databses into one dict.
				concat_dict[key] = np.append( concat_dict[key], tmp_dict[key] )

			print( "Completed... ", "\n" )


		df = pd.DataFrame()
		# Add each header as a column of the dataframe
		for key in concat_dict.keys():
			print( key, "  ", len( concat_dict[key] ) )
			df[f"{key}"] = concat_dict[key]

		df.to_csv( f"{self.dibs_mfib_csv}", index = False )

		return df


	def fuzz_db( self ):
		"""
		FuzzDB is a database of fuzzy protein complexes.
		Contains the PDB IDs, Uniprot IDs, and Uniprot boundaries of the disordered proteins only.
		Remove entries which do not have a PDB associated.
		For entries with >1 PDBs, split them into multiple entries in the output dataframe.
		"""

		# Load the tsv file as a dataframe
		data = pd.read_csv( self.fuzzdb, sep="\t" )
		print( "Total entries in Fuzzdb = ", len( data ) )

		# Create a dict to hold the relevant fields.
		fuzz_dict = {i:[] for i in ["PDB ID", "Asym ID1", "Asym ID2", \
		"Auth Asym ID1", "Auth Asym ID2", "Uniprot accession1", "Uniprot accession2", "Uniprot boundary1"] }

		logger = {i:[[], 0] for i in ["no_PDB", "entry_count"]}

		# Loop over all the entries in the file
		for i in range( len( data ) ):
			if pd.isna( data.iloc[i,10] ):
				logger["no_PDB"][0].append( data.iloc[i,10] )
				logger["no_PDB"][1] += 1

			# Only considering entries that have a pdb reference in Fuzzdb
			else:
				logger["entry_count"][0].append( data.iloc[i,10] )
				logger["entry_count"][1] += 1
				# In case an entry has >1 PDB, split them into multiple entries.
				# Multiple entries are separated by a ",".
				for j in str( data.iloc[i,10] ).split( "," ):
					fuzz_dict["PDB ID"            ].append( j.strip().split( ":" )[1] )
					fuzz_dict["Auth Asym ID1"     ].append( "None" )
					fuzz_dict["Auth Asym ID2"     ].append( "None" )
					fuzz_dict["Asym ID1"          ].append( "None" )
					fuzz_dict["Asym ID2"          ].append( "None" )
					fuzz_dict["Uniprot accession1"].append( data.iloc[i,1] )
					fuzz_dict["Uniprot accession2"].append( "None" )
					fuzz_dict["Uniprot boundary1" ].append( data.iloc[i,5] )

		print("Entries not having PDB structure = ",        logger["no_PDB"][1]  )
		print("Total (unsplit) entries = ",        logger["entry_count"][1]  )
		
		# Write all PDB IDs (comma separated) to a txt file.
		# Used to fetch entries from PDB. Uncomment below if required.
		# if self.main:
		# 	with open("Fuzzdb_pdbs.txt", "w") as w:
		# 		[w.writelines(str(i)+",") for i in fuzz_dict["PDB ID"]]


		with open( f"{self.base_path}Logs_Fuzzdb.txt", "w" ) as w:
			w.writelines( "Logs for the Fuzzdb database \n" )		
			for key in logger.keys():
				w.writelines(str(key) + "\n")
				[w.writelines(str(i) + "\n") for i in logger[key][0]]
				w.writelines( "Count = " + str(logger[key][1]) + "\n \n" )
		
		fuzz_db = pd.DataFrame()
		for key in fuzz_dict.keys():
			fuzz_db[key] = fuzz_dict[key]

		print( "Total obtained entries = ", len( fuzz_db ) )

		return fuzz_db


	def pdb_tot_cdr( self ):
		"""
		PDBtot and PDBcdr are datasets containing disordered proteins in complexes.
		Created by developers of FuzzPred.
		PDBtot --> disordered protein is either DOR or DDR but not both.
		PDBcdr --> disordered protein can be both DOR and DDR in different complexes.
		Contains the PDB ID, Chain ID and Uniprot ID for the disordered protein only.
		"""

		fuzzpred_dict = {i:[] for i in ["PDB ID", "Asym ID1", "Asym ID2", \
		"Auth Asym ID1", "Auth Asym ID2", "Uniprot accession1", "Uniprot accession2", "Uniprot boundary1"] }

		pdb_tot_cdr = []
		db = pd.read_excel(self.pdb_tot)
		print( "PDB TOT --> ", len( db["Complex (PDB ID)"] ) )
		entry_count = 0
		for i in range(len(db)):
			# Skip entries for which no PDB exists.
			if db["Complex (PDB ID)"][i] == db["Complex (PDB ID)"][i]:
				entry_count += 1
				
				pdb_chain = db["Complex (PDB ID)"][i].strip().split( " " )
				idr_pdb_dict = { i.split( "_" )[0]:[] for i in pdb_chain }
				[ idr_pdb_dict[i.split( "_" )[0]].append( i.split( "_" )[1] ) for i in pdb_chain ]

				for key in idr_pdb_dict.keys():
					pdb_tot_cdr.append( key )
					fuzzpred_dict["PDB ID"            ].append( key )
					fuzzpred_dict["Auth Asym ID1"     ].append( ",".join( idr_pdb_dict[key] ) )
					fuzzpred_dict["Auth Asym ID2"     ].append( "None" )
					fuzzpred_dict["Asym ID1"          ].append( "None" )
					fuzzpred_dict["Asym ID2"          ].append( "None" )
					fuzzpred_dict["Uniprot accession1"].append( db["Uniprot ID"][i] )
					fuzzpred_dict["Uniprot accession2"].append( "None" )
					fuzzpred_dict["Uniprot boundary1" ].append( str( db["Start (Uniprot)"][i] ) + "-" + str( db["Stop (Uniprot)"][i] ) )

		print( "Total (unsplit) entries in PDB TOT = ", entry_count )
		pdb_tot_entries = len( fuzzpred_dict["PDB ID"] )
		print( "Total entries in PDB TOT = ", pdb_tot_entries )
		entry_count_dor, entry_count_ddr = 0, 0
		db = pd.read_excel(self.pdb_cdr)
		print( "PDB CDR --> ", len( db["Start (Uniprot)"] ) )
		for field in ["Complex (PDB ID) DOR", "Complex (PDB ID) DDR"]:
			for i in range(len(db)):
				# Skip entries for which no PDB exists.
				if db[field][i] == db[field][i]:
					if "DOR" in field:
						entry_count_dor += 1
					else:
						entry_count_ddr += 1

					pdb_chain = db[field][i].strip().split( " " )				
					idr_pdb_dict = { i.split( "_" )[0]:[] for i in pdb_chain }
					[ idr_pdb_dict[i.split( "_" )[0]].append( i.split( "_" )[1] ) for i in pdb_chain ]
					
					for key in idr_pdb_dict.keys():
						pdb_tot_cdr.append( key )
						fuzzpred_dict["PDB ID"            ].append( key )
						fuzzpred_dict["Auth Asym ID1"     ].append( ",".join( idr_pdb_dict[key] ) )
						fuzzpred_dict["Auth Asym ID2"     ].append( "None" )
						fuzzpred_dict["Asym ID1"          ].append( "None" )
						fuzzpred_dict["Asym ID2"          ].append( "None" )
						fuzzpred_dict["Uniprot accession1"].append( db["Uniprot ID"][i] )
						fuzzpred_dict["Uniprot accession2"].append( "None" )
						fuzzpred_dict["Uniprot boundary1" ].append( str( db["Start (Uniprot)"][i] ) + "-" + str(db["Stop (Uniprot)"][i] ) )

		print( f"Total (unsplit) entries in PDB CDR: DOR --> = {entry_count_dor} \t DDR --> = {entry_count_ddr} " )
		print( "Total entries in PDB TOT and CDR = ", len( fuzzpred_dict["PDB ID"] ) )
		with open( f"{self.base_path}Logs_PDBtot_cdr.txt", "w" ) as w:
			w.writelines( f"Total (unsplit) entries in PDB TOT = {entry_count}\n" )
			w.writelines( f"Total entries in PDB TOT = {pdb_tot_entries}\n" )
			w.writelines( f"Total (unsplit) entries in PDB CDR: DOR --> = {entry_count_dor} \t DDR --> = {entry_count_ddr}\n" )
			w.writelines( f"Total entries in PDB TOT and CDR = {len( fuzzpred_dict['PDB ID'] )}\n" )

		fuzzpred = pd.DataFrame()
		for key in fuzzpred_dict.keys():
			print(key, " --> ", len(fuzzpred_dict[key]))
			fuzzpred[key] = fuzzpred_dict[key]
		fuzzpred.to_csv( self.pdb_tot_cdr_csv, index = False )

		return fuzzpred
	# 【中文解析-重点逻辑】forward 通常串联“读取输入 -> 清洗/映射 -> 导出中间/最终文件”，是该脚本最关键的数据流水线节点。
	def forward( self ):
		"""Extract PDB IDs and associated info from the five datasets 
		on structures of IDR complexes. Combine them into one dataframe/CSV file. 
		""" 
	 
		if not os.path.exists( self.dibs ):
			raise Exception( "DIBS database file not found in database/raw/..." )
		
		if not os.path.exists( self.mfib ):
			raise Exception( "MFIB database file not found in database/raw/..." )
		
		if not os.path.exists( self.fuzzdb ):
			raise Exception( "FuzzDB database file not found in database/raw/..." )
		
		if not os.path.exists( self.pdb_tot ):
			raise Exception( "PDBtot dataset file not found in database/raw/..." )
		
		if not os.path.exists( self.pdb_cdr ):
			raise Exception( "PDBcdr dataset file not found in database/raw/..." )

		if os.path.exists( f"{self.base_path}{self.output_file}" ):
			print( f"{self.output_file} already exists..." )
			df = pd.read_csv( f"{self.base_path}{self.output_file}" )
			print( "Total entries = ", len( df["PDB ID"] ) )

		else:
			dibs_mfib = self.dibs_mfib( dibs = True, mfib = True)
			print("\nDIBS MFIB done...")
			print("<================================>\n")
			fuzzdb    = self.fuzz_db()
			print("\nFuzzDB done...")
			print("<================================>\n")
			fuzzpred  = self.pdb_tot_cdr()
			print("\nPDB TOT CDR done...")
			print("<================================>\n")

			dataset = {i:[] for i in ["PDB ID", "Asym ID1", "Asym ID2", \
			"Auth Asym ID1", "Auth Asym ID2", "Uniprot accession1", "Uniprot accession2", "Uniprot boundary1"] }
			
			dataset = pd.concat( [dibs_mfib, fuzzdb, fuzzpred], axis = 0 )

			dataset.to_csv( f"{self.base_path}{self.output_file}", index = False )
			print("Total entries in the combined dataset = ", len( dataset["PDB ID"] ))
	 


##########################################################################
#------------------------------------------------------------------------#
class The3Muskteers():
	"""Get the list of PDBs from curated sequence databases on disordered proteins.
	The Uniprot IDs of sequences in these databases are used to query the PDB for 
	structures associated with the sequence. 
	DIBS, MFIB, FuzDB, PDBtot, PDBcdr are considered here.  
	"""

	def __init__( self, dir_name, cores ):
		"""
		Constructor. 
		The raw files (self.disprot, self.ideal), etc 
		are downloaded as is from the web servers of the databases. 
		The output_file containts a list of PDBs associated with the sequences from these databases.
		"""

		self.base_path = dir_name # "./input_files/"
		self.disprot_path = "./raw/DisProt release_2023_12 with_ambiguous_evidences.tsv"
		self.disprot_csv = f"{self.base_path}DisProt.csv"
		self.ideal_path = "./raw/IDEAL_17Apr24.xml"
		self.ideal_csv = f"{self.base_path}IDEAL.csv"
		self.mobidb_dir = "./raw/MobiDB/"
		self.mobidb_csv = f"{self.base_path}MobiDB.csv"
		self.mobidb_batch_size = 1000      # No. of entries per batch to be downloaded.
		self.max_cores = cores
		# self.mobidb_download_cores = 5 if self.max_cores>5 else self.max_cores   # For batch downloading MobiDB data.
		self.min_len = 7
		self.max_trials = 15  # max no. of attempts to download.
		self.wait_time = 20
		self.output_file = f"{self.base_path}Merged_DisProt_IDEAL_MobiDB.csv"

		# csv file containing PDB IDs obtained from DIBS, MFIB, Fuzdb, PDBtot and PDBcdr
		self.dibs_mfib_fdb_totcdr_file = f"{self.base_path}Merged_DIBS_MFIB_Fdb_PDBtot-cdr.csv"
		# File containng all unique PDB IDs from all the databases.
		self.merged_pdbs_file = f"{self.base_path}Merged_PDB_IDs.txt"
	# 【中文解析-重点逻辑】forward 通常串联“读取输入 -> 清洗/映射 -> 导出中间/最终文件”，是该脚本最关键的数据流水线节点。
	def forward( self ):
		"""
		Extract PDB IDs from the three sequence databases on IDRs using the Uniprot IDs of sequences. 
		Combine them into one dataframe/CSV file. 
		""" 

		if not os.path.exists( self.disprot_path ):
			raise Exception( "DisProt database file not found in database/raw/..." )
		
		if not os.path.exists( self.ideal_path ):
			raise Exception( "IDEAL database file not found in database/raw/..." )
		
		print("\n<================================>\n")
		if not os.path.exists( self.disprot_csv ):
			print( "Creating DisProt csv file..." )
			self.disprot()
			print( "\nDisProt done\n" )
			print("<================================>\n")
		else:
			print( "Disprot csv file already exists..." )
			df = pd.read_csv( self.disprot_csv )
			print( "Total entries in DisProt = ", len( df["Disprot ID"] ), "\n" )
			print( "Total unique entries in Disprot = ", len( pd.unique( df["Disprot ID"] ) ) )
		
		if not os.path.exists( self.ideal_csv ):
			print( "Creating IDEAL csv file..." )
			self.ideal2()
			print( "\nIDEAL done\n" )
			print("<================================>\n")
		else:
			print( "IDEAL csv file already exists..." )
			df = pd.read_csv( self.ideal_csv )
			print( "Total entries in IDEAL = ", len( df["IDP ID"] ), "\n" )
		
		if not os.path.exists( self.mobidb_csv ):
			print( "Creating MobiDB csv file..." )
			self.mobidb()
			print( "\nMobiDB done\n" )
			print("<================================>\n")
		else:
			print( "MobiDB csv file already exists...\n" )
			df = pd.read_csv( self.mobidb_csv )
			print( "Total entries in MobiDB = ", len( df["Uniprot ID"] ), "\n" )
			print( "Total unique entries in MobiDB = ", len( pd.unique( df["Uniprot ID"] ) ), "\n" )

		print( "Now fetching all PDB IDs for Uniprot IDs obtained from DisProt/IDEAL/MobiDB...\n" )
		if not os.path.exists( self.output_file ):
			self.all_pdbs_from_all_uniprots()

			df1 = pd.read_csv( self.dibs_mfib_fdb_totcdr_file )
			df2 = pd.read_csv( self.output_file )
			unique_pdbs = pd.unique( list( df1["PDB ID"] ) + list( df2["PDB ID"] ) )
			with open( self.merged_pdbs_file, "w" ) as w:
				w.writelines( ",".join( unique_pdbs ) )
		else:
			print( "Output file already exists..." )
			df1 = pd.read_csv( self.dibs_mfib_fdb_totcdr_file )
			df2 = pd.read_csv( self.output_file )
			
			unique_pdbs = pd.unique( list( df1["PDB ID"] ) + list( df2["PDB ID"] ) )
			with open( self.merged_pdbs_file, "w" ) as w:
				w.writelines( ",".join( unique_pdbs ) )



	def disprot( self ):
		""" 
		DisProt Parser
		Parse the DisProt database tsv file to extract relevant info.
		"""
		 
		df = pd.read_csv( self.disprot_path, sep = "\t" )

		tic = time.time()
		disprot_holocron = { "Disprot ID":[], "Uniprot ID":[], "Disorder regions":[] }
		for i, id_ in enumerate( df["disprot_id"] ):
			disprot_holocron["Disprot ID"].append( id_ )
			disprot_holocron["Uniprot ID"].append( df["acc"][i] )

			start = df["start"][i]
			end = df["end"][i]	
			disprot_holocron["Disorder regions"].append( f"{start}-{end}" )

			# print( f"{i} --> Done for DisProt ID: {id_} --------------------------" )

		toc = time.time()
		print( "Total entries in Disprot = ", len( disprot_holocron["Disprot ID"] ) )
		print( "Total unique entries in Disprot = ", len( set( disprot_holocron["Disprot ID"] ) ) )

		df_new = pd.DataFrame()
		for key in disprot_holocron.keys():
			df_new[key] = disprot_holocron[key]
		df_new.to_csv( self.disprot_csv, index = False )

		print( f"Total time taken = {toc-tic} seconds" )
		print( "May the Force be with you..." )



	def get_obsolete_uni_id( self, uni_ids ):
		"""  
		Check if the uniprot ID is obsolete or not.
		Input:
		----------
		uni_ids --> comma separated string of uniprot IDs.

		Returns:
		----------
		2 tuples, one of active IDs and another of obsolete IDs.  
		"""    
	 
		uni_ids = uni_ids.split( "," )
		obsolete, active = [], []
		num_uni_ids = len( uni_ids )

		# cores = max_cores if num_uni_ids > self.max_cores else num_uni_ids
		with Pool( self.max_cores ) as p:
			for result in p.imap_unordered( 
								partial( get_uniprot_seq, 
									max_trials = self.max_trials, 
									wait_time = self.wait_time, 
									return_id = True ), 
								uni_ids ):
				id_, seq = result
				if seq == []:
					obsolete.append( id_ )
				else:
					active.append( id_ )

		return ",".join( active ), ",".join( obsolete )


	def ideal2( self ):
		""" 
		IDEAL Parser
		Extract Uniprot ID and other relevant info from IDEAL xml file.
		Considering all disordered regions for verified ProS.
		"""

		root = ET.parse( self.ideal_path ).getroot()
		
		tic = time.time()
		ideal_holocron = {i:[] for i in ["IDP ID", "Uniprot ID", "Disorder regions", "Uniprot ID (obsolete)"]}
		
		for i, tag in enumerate( root.findall( f"IDEAL_entry" ) ):
			verified = False
			idp_id = tag.find(f"idp_id").text

			# if idp_id == "IID00165":
			# Get all uniprot IDs for the entry.
			uni_ids = ",".join( [uni_id.text for uni_id in tag.findall(f"General/uniprot") ] )

			# Get all verified ProS disorder regions.
			pros_type = [t_.text for t_ in tag.findall( "Function/pros_type" )]
			start = [ds.text for ds in tag.findall( "Function/disorder_location/disorder_region_start" )]
			end = [de.text for de in tag.findall( "Function/disorder_location/disorder_region_end" )] 

			tmp_dr = []

			for idx, type_ in enumerate( pros_type ):
				if type_ == "verified" and start[idx] != None and end[idx] != None:
					# print( f"IDP ID {idp_id} contains a verified ProS..." )
					verified = True
					tmp_dr.append( f"{start[idx]}-{end[idx]}" )

			# Selecting only disorder regions for verified ProS.
			if verified:
				ideal_holocron["IDP ID"].append( idp_id )
				
				# Segregate the obsolete and active uniprot IDs.
				active, obsolete = self.get_obsolete_uni_id( uni_ids )

				# active, obsolete = uni_ids, ""
				ideal_holocron["Uniprot ID"].append( active )
				ideal_holocron["Uniprot ID (obsolete)"].append( obsolete )		
				
				ideal_holocron["Disorder regions"].append( ",".join( tmp_dr ) )
			# print( f"{i} --> Done for IDP ID: {idp_id} --------------------------" )

		df = pd.DataFrame()
		for key in ideal_holocron.keys():
			df[key] = ideal_holocron[key]
		df.to_csv( self.ideal_csv, index = False )

		print( "Total entries in IDEAl = ", len( df["IDP ID"] ) )

		toc = time.time()
		print( f"Total time taken = {( toc-tic )/60} minutes" )


	def mobidb_parallel( self, partition, category ):
		"""
		Download a subset (partition) of MobiDB entries.
		Input:
		----------
		partition --> contains the batch to be downloaded; contains the limit and skip values.
		category --> specifies the category of data to be downloaded from MobiDB.

		Returns:
		----------
		dict containing Uniprot ID, disordered regions, and their annotation. 
		"""

		skip, limit = partition
		current = f"{skip}-{skip+limit}"
		# print( f"\nPartition {current}..." )

		# Randomly delay downloading different partitions by rn seconds; rn ~ U(a,b).
		rn = random.uniform( 1, 2 )
		time.sleep( rn )
		
		# Considering only disorder regions with a priority consensus.
		fields_included = ["curated-disorder-priority", "curated-lip-priority", "homology-disorder-priority", "homology-lip-priority"]
		
		download_url = f"https://mobidb.org/api/download?format=json&limit={limit}&skip={skip}&{category}=exists"
		
		"""
		Download the batch and check if the first entry is loadable or not (low pass test).
		Loop over all entries and load the JSON dict.
			Obtain the Uniprot ID and Uniref ID. 
			Also obtain the priority disorder, lip regions from curated and homology evidence.
		If it fails at any point, redownload the batch.
		For each redownload, start from the last unsuccessfully loaded entry instead of completly restarting again.
		"""

		# t1 = time.time()
		last_outpost = 0
		data_dict = {}
		# for i,entry in enumerate( response.iter_lines() ):
		for trial in range( self.max_trials ):
			response = send_request( download_url, _format = None, max_trials = self.max_trials )

			try:
				data = json.loads( next( response.iter_lines() ) )
				
				all_entries = [entry for entry in response.iter_lines()]
				jump_start = last_outpost
				jump_end = len( all_entries )

				for i,entry in enumerate( all_entries[jump_start:jump_end] ):
					data = json.loads( entry )

					i += jump_start  
					data_dict[f"{skip+i}"] = {key:[] for key in ["Uniprot ID", "Disorder regions", "Annotation"]}

					data_dict[f"{skip+i}"]["Uniprot ID"].append( data["acc"] )
					for key in fields_included:
						if key in data.keys():
							data_dict[f"{skip+i}"]["Disorder regions"].extend( [f"{x[0]}-{x[1]}" for x in data[key]["regions"]] )
							data_dict[f"{skip+i}"]["Annotation"].append( key )
					last_outpost += 1
				break

			except Exception as e:
				if trial != self.max_trials-1:
					# print( f"Handling the error in \t Error in partition {current}...\n" )
					time.sleep( trial*1 )
					continue
				else:
					print( "Nooooooooooooooooooo....\n" )
					exit()

		# t2 = time.time()
		# print( f"\nCompleted for partition {current} \t|\t Time taken = {( t2-t1 )/60} minutes...\n" )
		return data_dict	


	def mobidb( self ):
		""" 
		MobiDB Parser
		Obtain Uniprot IDs, disorder region and annotation from MobiDB
		This calls mobidb_parallel to download MobiDB entries in batches in parallel. 
		"""   

		tim = time.time()
		mobidb = {i:[] for i in ["Uniprot ID", "Disorder regions", "Annotation"]}
		categories_included = ["curated-disorder-merge", "curated-lip-merge", "homology-disorder-merge", "homology-lip-merge"]

		w_log = open( f"{self.base_path}Logs_MobiDB.txt", "w" )
		w_log.writelines( "<--------------------- Logs: MobiDB --------------------->\n" )
		
		for category in categories_included:
			tic = time.time()
			print( f"Downloading data for category: {category}" )
			count_url = f"https://mobidb.org/api/count?format=json&{category}=exists"

			response = send_request( count_url, _format = "json" )
			total = response["n"]

			print( f"Total entries in Category - {category} = {total}" )
			w_log.writelines( f"Category: {category}\n" )
			w_log.writelines( f"Total entries on MobiDB = {total}\n" )

			# Create batches to be downloaded.
			# partitions = np.arange( 0, total, self.mobidb_batch_size )
			if total < self.mobidb_batch_size:
				skip, limit = [0], [total]
			else:
				skip = np.arange( 0, total, self.mobidb_batch_size )
				limit = [self.mobidb_batch_size for i in range( len( skip )-1 )]
				last = ( total - skip[-1] )
				limit = np.append( limit, last )

			partitions = list( zip( skip, limit ) )
			# print( f"Created the following partitions: \n {partitions}\n-----------------------" )
			# print( f"Created the following partitions: \n-----------------------" )

			# cores = 1 if len( partitions ) == 1 else self.mobidb_download_cores
			with Pool( 10 ) as p:
				results = tqdm.tqdm( p.imap_unordered( 
											partial( self.mobidb_parallel, category = category ), partitions ), 
										total = len( partitions ) )

				for result in results:
					for key in result.keys():
						if result[key]["Disorder regions"] != []:
							mobidb["Uniprot ID"].append( ",".join( result[key]["Uniprot ID"] ) )
							mobidb["Disorder regions"].append( ",".join( result[key]["Disorder regions"] ) )
							mobidb["Annotation"].append( ",".join( result[key]["Annotation"] ) )

			toc = time.time()
			w_log.writelines( "Total entries obtained so far = " + str( len( mobidb["Uniprot ID"] ) ) )
			w_log.writelines( f"\nTime taken = {( toc-tic )/60} minutes" )
			w_log.writelines( "\n-------------------------------------\n" )
			print( f"Time taken = {( toc - tic )/60} minutes" )
			print( f"Completed for {category}... \t {len( mobidb['Uniprot ID'] )}\n\n" )


		w_log.writelines( f"Total entries in MobiDB = {len( mobidb['Uniprot ID'] )}\n" )
		w_log.writelines( f"Total unique entries in MobiDB = {len( pd.unique( mobidb['Uniprot ID'] ) )}\n\n" )
		df = pd.DataFrame()
		for key in mobidb:
			print( f"{key} \t {len( mobidb[key] )}" )
			df[key] = mobidb[key]

		df.to_csv( self.mobidb_csv, index = False )
		tom = time.time()
		time_ = ( tom-tim )/3600
		w_log.writelines( f"Total time taken = {time_} hours..." )
		print( f"Total time taken = {time_} hours" )


	
	def get_all_pdb_ids( self, method ):
		""" 
		Deprecated function, no longer used.
		Obtain all PDB IDs available on PDB.
		By manually downloading lists of available PDB IDs.
		Using Biopython module to get all PDB IDs.
		Using the PDB REST API to fetch all available PDB IDs.
		"""
		
		all_pdbs = []
		if method == "manual":	
			# Get all PDB IDs present on PDBs(15 Apr, 2023).
			for num in ["1-75000", "75001-150000", "150001-199296"]:
				with open( f"{self.path}PDB_IDs/PDB_IDs_{num}.txt" ) as f:
					all_pdbs +=  str( f.readlines() ).replace( "'", "" ).replace( "[", "" ).replace( "]", "" ).split( "," )
		
		else:
			if os.path.exists( f"{self.path}/All_PDB_IDs_{self.date}" ):
				with open( f"{self.path}/All_PDB_IDs_{self.date}.txt", "r" ) as f:
					all_pdbs = f.readlines()[0].split( "," )
			
			else:
				if method == "bio":
					pdb = PDB.PDBList( server = "ftp://ftp.wwpdb.org" )
					all_pdbs = pdb.get_all_entries()[1:]

					all_pdbs = ",".join( all_pdbs )

				elif method == "pdb":
					url = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
					data = send_request( url )
					all_pdbs = ",".join( data )

				with open( f"{self.path}/All_PDB_IDs_{self.date}.txt", "w" ) as w:
					w.writelines( all_pdbs )

		return all_pdbs


	def get_all_unique_uniprot_ids( self ):
		"""
		Obtain all unique uniprot IDs extracted from DisProt/IDEAL/MobiDB.
		"""
		disprot = pd.read_csv( self.disprot_csv )
		ideal = pd.read_csv( self.ideal_csv )
		mobidb = pd.read_csv( self.mobidb_csv )

		disprot_uni = ",".join( disprot["Uniprot ID"] )
		ideal_uni = ",".join( ideal["Uniprot ID"] )
		mobidb_uni = ",".join( mobidb["Uniprot ID"] )

		all_uni = pd.unique( ( disprot_uni + ideal_uni + mobidb_uni ).split( "," ) )
		print( "All unique Uniprot IDs = ", len( all_uni ) )

		return all_uni


	def get_PDBs_for_uniprot_id( self, uni_id ):
		"""
		Obtain PDB IDs corresponding to a set of Uniprot IDs. 
		All PDB IDs related to each Uniprot ID are obtained using the PDB REST API.
		
		Input:
		----------
		uni_id --> UniProt accession ID.

		Returns:
		----------
		[Uniprot ID, list of PDB IDs]
		"""
		# num, uni_id = labelled_uni_ids
		
		# for i in range( self.max_trials ):
		pdbs = get_PDB_from_Uniprot_pdb_api( uni_id, 
											max_trials = self.max_trials, 
											wait_time = self.wait_time )
			# if pdbs == None:
			# 	pdbs = None
			# 	break
			
			# elif len( pdbs ) != 0:
			# 	break
			
			# else:
			# 	time.sleep( i*1 )
			# 	continue

		return [uni_id, pdbs]



	def check_for_disorder( self, uniprot_ids ):
		"""
		Deprecated function, no longer used.
		A low pass check for disorder.
		Identify IDRs by checking if uniprot ID exists on DIsprot/IDEAL.
		Input: uniprot_ids --> all uniprot IDs obtained from PDB for a PDB ID.
		"""

		disorder = []
		for uni_id in uniprot_ids:
			disorder_regions = find_disorder_regions( self.disprot_path, self.ideal_path, uni_id, min_len = self.min_len )
			if disorder_regions == []:
				disorder.append( False )
			else:
				disorder.append( True )

		if all( disorder ):
			return True
		else:
			return False


	def filter_PDBs( self, pdb_id ):
		""" 
		Deprecated function, no longer used.
		Download PDB files containing >1 protein chains.
		Input:
		----------
		pdb_id --> PDB ID for the entry.

		Returns:
		----------
		None
		""" 

		num, pdb_id = pdb_id[0], pdb_id[1]

		print( f"Checking for entry {num}_{pdb_id}..." )
		
		entity_ids, asym_ids, auth_asym_ids, uniprot_ids, _ = from_pdb_rest_api_with_love( pdb_id )

		# Get all chain IDs in PDB.
		all_uniprot_ids = [uni_id for e in entity_ids for uni_id in uniprot_ids[e] ]

		disorder_present = self.check_for_disorder( all_uniprot_ids )

		# Identify if disordered proteins are present.
		if not disorder_present:
			print( f"{num}_{pdb_id} has no disordered regions...\n" )
			return ""

		else:
			print( "\n" )
			return pdb_id


	def the_way_of_pdb( self ):
		"""
		Deprecated function, no longer used.
		Starting from all PDB IDs in PDB, select those:
			not monomers.
			at least 1 chain is disordered.
			has a mapping on SIFTS.
		"""
		all_pdbs = self.get_all_pdb_ids_Bio()

		checkpoints = np.arange( 0, len( all_pdbs ), 5000 )
		remainder = len( all_pdbs ) - checkpoints[-1]
		last_batch = checkpoints[-1] + remainder
		checkpoints = np.append( checkpoints, last_batch )
		
		complexes = []
		for start, end in zip( checkpoints, checkpoints[1:] ):
			jump_start = start
			jump_end = end 
			num = np.arange( jump_start, jump_end, 1 )
			labelled_pdb = list( zip( num, all_pdbs[jump_start:jump_end] ) )
			try:
				with Pool( self.max_cores ) as p:
					results = p.map( self.filter_PDBs, labelled_pdb )
			except:
				with ProcessPoolExecutor() as executor:
					complexes = list( executor.map( self.filter_PDBs, labelled_pdb ) )
			complexes = np.append( complexes, results )
			complexes = complexes[np.where( complexes != "" )]

		df = pd.DataFrame()
		df["PDB ID"] = complexes
		df.dropna( axis = 0, inplace = True)
		df.to_csv( f"All_complex_PDBs.csv", index = False )


	def all_pdbs_from_all_uniprots( self ):
		"""
		Obtain all PDB IDs corresponding to all the Uniprot IDs from DisProt/IDEAL/MobiDB and save as a csv file.
		"""

		if os.path.exists( f"{self.base_path}{self.output_file}" ):
			print( f"{self.output_file} already exists..." )
		
		else:
			tic = time.time()
			no_PDBs, pdb_ids = [], []
			all_uniprot_ids = self.get_all_unique_uniprot_ids()		
			
			counter = 0
			with Pool( self.max_cores ) as p:
				for result in tqdm.tqdm( 
										p.imap_unordered( 
														self.get_PDBs_for_uniprot_id, 
														all_uniprot_ids ), 
										total = len( all_uniprot_ids ) ):

					if len( result[1] ) == 0:
						no_PDBs.append( result[0] )
					else:
						# Multiple PDBs can exist for a Uniprot ID.
						pdb_ids.extend( result[1] )

					if counter != 0 and counter%50000 == 0:
						print( f"Total PDBs obtained so far = {len( pdb_ids )} \t {len( pd.unique( pdb_ids ) )}" )

					counter += 1
				
			# print( f"Completed for batch {start}-{end}..." )
			print( f"Total PDBs obtained = {len( pdb_ids )}" )
			print( f"Total unique PDBs obtained so far = {len( pd.unique( pdb_ids ) )}\n" )


			df = pd.DataFrame()
			df["PDB ID"] = pd.unique( pdb_ids )
			df.to_csv( f"{self.base_path}{self.output_file}", index = False )

			toc = time.time()

			# Log the Uniprot IDs for which no PDB IDs were obtained.
			with open( f"{self.base_path}Logs_noPDB_in_Uniprot.txt", "w" ) as w:
				w.writelines( f"Total = {len(no_PDBs)}\n" )
				[w.writelines( f"{id_}," ) for id_ in no_PDBs]

			proc = subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE )
			system = proc.communicate()[0]

			proc = subprocess.Popen( "date", shell = True, stdout = subprocess.PIPE )
			sys_date = proc.communicate()[0]

			with open( f"{self.base_path}Logs_sequence_databases.txt", "w" ) as w:
				w.writelines( f"Created on: System = {system} \t Date = {sys_date}\n" )
				w.writelines( f"Time taken to get all PDB IDs = {( toc-tic )/3600} hours\n\n" )
				w.writelines( f"No PDBs for Uniprot ID = {len(no_PDBs)}\n" )
				w.writelines( f"Total Unique Uniprot IDs = {len(all_uniprot_ids)}\n" )

			print( f"Time taken to get all PDB IDs = {( toc-tic )/3600} hours" )
			print( "Total Uniprot IDs = ", len( all_uniprot_ids ) )
			print( "Total PDB IDs obtained = ", len( df["PDB ID"] ) )


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parse a) sequence and b) structure databases to get two corresponding "+
								  "lists of PDBs (Merged_*.csv) containing IDRs in complex.") 

	parser.add_argument( '--max_cores', '-c', dest="c", help = "No. of cores to be used.", type = int, required=False, default=10 )
	
	t0 = time.time()
	database_path = f"../database/"
	dir_name = "input_files/"

	if not os.path.exists( f"{database_path}{dir_name}" ):
		os.makedirs( f"{database_path}{dir_name}" )

	os.chdir( database_path )

	cores = parser.parse_args().c

	# 1. Get the PDB IDs of structures of complexes with IDRs in curated IDR structure databases.
	# These are in the Merged_DIBS*.csv file. 
	TheIlluminati( dir_name ).forward()

	# 2. Get additional PDBs from sequences of IDRs from curated IDR sequence databases.
	# These are in the Merged_Disprot*.csv file. 
	The3Muskteers( dir_name, cores ).forward()

	t = time.time()
	print( f"Total time taken = {( t-t0 )/3600} hours..." )
	print( "May the Force be with you..." )

"""
【中文解析-可演化改动建议】
1) 将硬编码路径迁移到统一配置文件（YAML/ENV）并加入路径合法性校验。
2) 为每个中间文件增加 schema 校验（字段名/类型/范围），降低跨脚本耦合导致的隐式错误。
3) 增加小样本 smoke test（例如 3~5 条记录）和数据快照测试，保障重构时行为一致。
4) 若面向预训练，建议输出 token 级别元信息（来源数据库、结构置信度、对齐覆盖率）便于构建多任务标签。
"""
