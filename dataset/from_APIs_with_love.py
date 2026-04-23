"""
【中文解析-模块总览】
- 中心功能：from_APIs_with_love.py 属于 Disobind 数据流水线脚本，用于数据抓取/清洗/映射/导出中的一个环节。
- 逻辑位置：该文件在“原始数据库 -> 统一样本定义 -> 训练/评估输入文件”链条中承担局部处理任务。
- 输入输出流水线：通常输入为数据库文件（csv/tsv/xlsx/json）、结构文件（pdb/cif）与序列标识（UniProt/PDB），输出为中间表、映射字典、模型可读输入（csv/h5/json/npy/fasta）。
- 可选项：多数脚本支持通过构造参数或命令行参数控制版本号、并行核数、路径和过滤阈值。
- 可视化能力：数据脚本本身通常不直接出图；如有图像分析一般在 analysis/ 目录统一完成。
- 数据格式与实验标签：核心监督标签通常是残基-残基接触图（contact map）或界面位点（interface residues）；本目录负责把外部异构数据规范化成论文实验使用的统一格式。
- 数据来源与论文逻辑：数据来源包含 DIBS/MFIB/DisProt/IDEAL/MobiDB/PDB/SIFTS 等；处理逻辑服务于“按二元复合体构建样本并对齐序列-结构关系”。
- 预训练扩展建议：若做进一步预训练，建议先扩展原始来源覆盖面、统一长度分桶策略、保留更多中间质控字段（置信度/覆盖率/冲突标记），再生成更大规模弱监督样本。
"""

"""
Helper functions to fetch data from web servers, APIs
####### ------>"May the Force serve u well..." <------#######
"""

############# One above all #############
##-------------------------------------##
import requests, wget, json, time, os, subprocess
from io import StringIO
import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

##########################################
##--------------------------------------##
# 【中文解析-关键函数/类入口】以下对象是本文件的主执行入口，承担数据转换主链路。
# 【中文解析-重点逻辑】send_request 负责外部数据源访问与失败重试，是上游数据稳定性的关键保障。
def send_request( url, _format = "json", max_trials = 10, wait_time = 5 ):
	"""
	Send a request to the server to fetch the data.
		For Httpresponse 404 returns "not_found".
		For Httpresponse 400 returns "bad_request".
	
	Input:
	----------
	url --> URL of the server from where to fetch the data.
	_format --> output format for the server reponse (json or text).
				If None, the response is returned.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
		This is a variant of the Exponential backoff algorithm.


	Returns:
	----------
	Currently, will return either of:
		_format = None: response object.
		_format = json: return a JSON dict.
		_format = text: return the response in text format.
	"""
	for trial in range( 0, max_trials ):
		try:
			response = requests.get( url )

			# Resource not found.
			if response.status_code == 404:
				if trial > ( max_trials/2 ):
					continue
				else:
					return "not_found"

			# Bad request.
			elif response.status_code == 400:
				if trial != ( max_trials - 1 ):
					continue
				else:
					return "bad_request"

			elif response.status_code == 200:
				if _format == None:
					return response
				elif _format == "json":
					return response.json()
				else:
					return response.text
				break
			
			else:
				raise Exception( f"--> Encountered status code: {response.status_code}\n" )
		except Exception as e:
			if trial != max_trials-1:
			# 	print( f"Trial {trial}: Exception {e} \t --> {url}" )
				continue
			else:
				return "not_found"


############### Uniprot API ##############
##--------------------------------------##
##########################################

############## Uniprot entry #############
##--------------------------------------##
def get_uniprot_entry( uni_id, max_trials = 10, wait_time = 5 ):
	"""
	Obtain entry details from Uniprot for a given uni_id.

	Input:
	----------
	uni_id --> Uniprot accession.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
		This is a variant of the Exponential backoff algorithm.

	Returns:
	----------
	data --> None for "not_found" or "bad_request".
			0 is returned if the Uniprot ID is obsolete (entryType inactive).
			Else return the JSON dict.
	"""
	url = f"https://rest.uniprot.org/uniprotkb/{uni_id}.json"

	data = send_request( url, _format = "json", max_trials = max_trials, wait_time = wait_time )
	if data == "not_found" or data == "bad_request":
		return None
	
	else:
		# data = response.json()
		if data["entryType"] == "Inactive":
			return 0
		else:
			return data


def get_uniprot_entry_name( uni_id, max_trials = 10, wait_time = 5 ):
	"""
	Read the Uniprot entry obtained as an xml file.

	Input:
	----------
	uni_id --> Uniprot accession.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	Name for the given Uni ID.
	For obsolete Uni ID None is returned.
	An empty list is returned if couldn't find Name for Uni ID.
	"""
	for trial in range( max_trials ):
		data = get_uniprot_entry( uni_id, max_trials = max_trials, wait_time = wait_time )
		
		if data == None:
			return None

		elif data == 0:
			return None
		
		else:
			if "recommendedName" in data["proteinDescription"].keys():
				prot_name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
			elif "submissionNames" in data["proteinDescription"].keys():
				prot_name = data["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
			
			# Try again in case it fails to fetch data, prot_name is an empty list.
			if prot_name == []:
				continue
			else:
				return prot_name

# print( get_uniprot_entry_name( "A2ARV4" ) ) # Q6LD08, P04150, Q8PJB5, Q6LD08, P0DJZ2
# exit()

############### Uniprot seq ##############
##--------------------------------------##
def get_uniprot_seq( uni_id, max_trials = 5, wait_time = 5, return_id = False ):
	"""
	Obtain Uniprot seq for the specified Uniprot ID.

	Input:
	----------
	uni_id --> Uniprot accession.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
	
	Returns:
	----------
	Sequence for the given Uni ID.
		Uni ID is returned if specified.
	An empty list will be returned if couldn't find the sequence for the given Uni ID.
	"""
	url = f"http://www.uniprot.org/uniprot/{uni_id}.fasta"
	
	data = send_request( url, _format = "text", max_trials = max_trials, wait_time = wait_time )
	if data == "not_found" or data == "bad_request":
		return [uni_id, []] if return_id else []
	
	else:
		seq_record = [str( record.seq ) for record in SeqIO.parse( StringIO( data ), 'fasta' )]

		if seq_record == []:
			return [uni_id, []] if return_id else []
		else:
			return [uni_id, seq_record[0]] if return_id else seq_record[0]


# seq = get_uniprot_seq("A1B602")
# print( len(seq), "\n", seq ) # A0A024B7W1
# exit()
# defaulters = []
# for seq in ['E1BQ43', 'I3LJZ9', 'B0V5X3', 'U6N325', 'U9XX47', 'D7XKZ3', 'A0A1X3LT86', 'U9Y6H3', 'D8EB41']:
# 	sequence = get_uniprot_seq( seq )
# 	print( sequence, "\n" )
# 	if len( sequence ) != 0:
# 		defaulters.append( seq )
# print( defaulters )
# print( len( defaulters ) )
# exit()


## Get all PDB IDS for a Uniprot ID using Uniprot API ##
#########--------------------------------------#########
def get_PDB_from_Uniprot_uni_api( uni_id ):
	"""
	Obtain all the PDB IDs corresponding to a Uniprot ID using Uniprot programmatic access.

	Input:
	----------
	uni_id --> Uniprot accession.
	
	Returns:
	----------
	List of PDB IDs for the given Uni ID.
	"""
	uni_id = "P68336"
	url = f"https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+accession:{uni_id}+AND+database:pdb&format=tsv&fields=xref_pdb"
	data = send_request( url, _format = "text" )
	if data == None:
		return None
	else:
		data = data.split( "\n" )[1].split( ";" )
		return data

# print( get_PDB_from_Uniprot_uni_api( "P01106" ) )
# exit()


#### Get all PDB IDS for a Uniprot ID using PDB API ####
#########--------------------------------------#########
def get_PDB_from_Uniprot_pdb_api( uni_id, max_trials = 10, wait_time = 5 ):
	"""
	Obtain all the PDB IDs corresponding to a Uniprot ID using PDB API.

	Input:
	----------
	uni_id --> Uniprot accession.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	List of PDB IDs for the given Uni ID.
	"""
	uni_id = uni_id.split( "-" )[0]
	url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_groups/{uni_id}"
	response = send_request( url, _format = None )

	if response == "not_found" or response == "bad_request":
		all_pdb_ids = []
	
	else:
		data = response.json()

		if "rcsb_group_container_identifiers" not in data.keys():
			all_pdb_ids = []

		else:
			all_pdb_ids = data["rcsb_group_container_identifiers"]["group_member_ids"]
			# PDB IDs contain only 4 characters + maybe chain ID. "{PDB_ID}_{Chain_ID}" e.g. "1A0N_A".
			all_pdb_ids = [id_[:4] for id_ in all_pdb_ids if "AF" not in id_]

	return all_pdb_ids


# print( get_PDB_from_Uniprot_pdb_api( "Q9KI21" ) ) # P68336, A0A6J2BJ57, B7Z1Q1, P63096, P50148
# exit()


############### 1 letter code #############
##--------------------------------------##
def name_to_symbol( aa ):
	"""
	Converts 3-letter amino acid names to symbols

	Input:
	----------
	aa --> 3-letter code for amino acid.

	Returns:
	----------
	1-letter code for amino acid.

	"""
	if aa == 'ASP':
	    return 'D'
	elif aa == 'GLU':
	    return 'E'
	elif aa == 'PHE':
	    return 'F'
	elif aa == 'LYS':
	    return 'K'
	elif aa == 'ASN':
	    return 'N'
	elif aa == 'GLN':
	    return 'Q'
	elif aa == 'ARG':
	    return 'R'
	elif aa == 'TRP':
	    return 'W'
	elif aa == 'TYR':
	    return 'Y'
	elif aa == 'PYL':
	    return 'O'
	elif aa == 'SEC':
	    return 'X'
	else:
	    return aa[0]


########### Download PDB Files ###########
##--------------------------------------##
def download( url, file_name, max_trials = 10, wait_time = 5 ):
	"""
	Download the file using requests and wget libraries.

	Input:
	----------
	url --> URL for the file to be downloaded.
	file_name --> name for the file to save downloaded content.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	success --> (bool) True if file downloaded without any error.
	"""
	try:
		response = requests.get( url )
		if response.status_code == 200:
			open( f"{file_name}", "wb" ).write( response.content )
			success = True
		else:
			success = False
	except:
		success = False

	return success



def pdb_valid( file_name, ext ):
	"""
	Check if a valid PDB file has been downloaded.
	PDB file is valid if:
		Can be read with Biopython Parser.
		Contains at least 1 model.
	Use PDB or MMCIF Parser as needed.

	Input:
	----------
	file_name --> Path for the downloaded PDB file.
	ext --> pdb or cif.

	Returns:
	----------
	True if PDB file is valid else False.
	"""

	try:
		if ext == "cif":
			models = MMCIFParser().get_structure( "cif", file_name )
			if len( models ) == 0:
				return False
			else:
				return True
			
		elif ext == "pdb":
			models = PDBParser().get_structure( "pdb", file_name )
			if len( models ) == 0:
				return False
			else:
				return True

	except:
		return False


def download_pdb( pdb_id, max_trials = 5, wait_time = 5, return_id = True ):
	"""
	Download the PDB entry in PDB format.
	A PDB ID can exist in PDB or CIF format.
		Try downloading PDB file first else download CIF file.
	If a download attempt fails, try redownloading until max_trials.
		pdb_id = None, if couldn't download in max_trials.

	Input:
	----------
	pdb_id --> PDB ID to be downloaded.
	lib --> specify the library to be used for downloading (requests or wget supported).
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
	return_id --> (bool) return thr pdb_id if True.

	Returns:
	----------
	"pdb"/"cif" --> label to specify if pdb or cif file was downloaded.
	pdb_id --> same as input.
	"""
	import warnings
	warnings.filterwarnings("ignore")
	
	pdb_id = pdb_id.lower()
	# try:
	for trial in range( max_trials ):
		try:
			url = f"https://files.rcsb.org/download/{pdb_id}.cif"

			file_name = f"./{pdb_id}.cif"
			success = download( url, file_name )
			if success:
					if pdb_valid( file_name, "cif" ):
						return ["cif", pdb_id] if return_id else "cif"

					elif trial != ( max_trials - 1 ):
						continue

					else:
						return [None, pdb_id] if return_id else None
			else:
				url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
				file_name = f"./{pdb_id}.pdb"
				success = download( url, file_name )
				if success:
					if pdb_valid( file_name, "pdb" ):
						return ["pdb", pdb_id] if return_id else "pdb"
			
					elif trial != ( max_trials - 1 ):
						continue
			
					else:
						return [None, pdb_id] if return_id else None

		except Exception as e:
			if trial == ( max_trials - 1 ):
				return [None, pdb_id] if return_id else None
			else:
				time.sleep( wait_time )
				continue


# print( download_pdb( "4v70" ) ) # 7a4j, 5mdz, 7s1k, 5o61, 4v6c
# exit()

################## PDBSWS ################
##--------------------------------------##
def call_pdbsws_rest( pdb ):
	"""
	Use PDBSWS to download PDB-Uniprot mapping.

	Input:
	----------
	pdb --> PDB ID to fetch mapping for.

	Returns:
	----------
	mapping --> PDB to Uniprot mapping in " " separated format.
	"""
	pdb = pdb.lower()
	url = f"http://www.bioinf.org.uk/servers/pdbsws/query.cgi?plain=1&qtype=pdb&id={pdb}&all=yes"
	data = send_request( url, _format = "text" )

	w = open( pdb + ".txt", "w" )
	w.writelines( data )
	w.close()

	with open( pdb + ".txt", "r" )as f:
		mapping = f.readlines()
	subprocess.call(["rm", f"{pdb}.txt"])
	# Initial few (0-5) lines are just headers, so ignoring them.
	return mapping[5:]
	


def get_pdbsws_mapping_dict( mapping, chain1, chain2 = None ):
	"""
	Get PDB to Uniprot mapped residue info from mapping obtained using PDBSWS.
	The file contains the following info in order:
		PDB_ID PDB_chain      PDB_no. PDB_res  PDB_pos    Uniprot_ID     Uniprot_res        Uniprot_pos

	Input:
	----------
	pdb --> PDB ID to be mapped.
	chain1 --> chain ID for which to obtain mapping.
	chain2 --> chain ID for which to obtain mapping.
		If chain2 == None, only get chain1 mapping.

	Returns:
	----------
	mapping_dict1 --> dict to store mapping info for chain1.
	mapping_dict2 --> dict to store mapping info for chain2.
	"""

	mapping_dict1 = {i:[] for i in ["pdb_seq", "pdb_pos", "uni_seq", "uni_pos"]}
	mapping_dict2 = {i:[] for i in ["pdb_seq", "pdb_pos", "uni_seq", "uni_pos"]}
	
	# Fetch the PDB seq, position, uniprot ID, seq, pos.
	for i in range(len(mapping)):
		line = " ".join( mapping[i].strip().split() ).split( " " )
		# Ignore if mapped PDB residue does not have Uniprot info.
		if len( line )>5:
			if len( line )>5 and line[1] == chain1:
				mapping_dict1["uni_id"] = line[5]
				mapping_dict1["uni_seq"].append( line[6] )
				mapping_dict1["uni_pos"].append( line[7] )
				mapping_dict1["pdb_seq"].append( name_to_symbol( line[3] ) )
				mapping_dict1["pdb_pos"].append( line[4] )

			elif chain2 != None and line[1] == chain2:
				mapping_dict2["uni_id"] = line[5]
				mapping_dict2["uni_seq"].append( line[6] )
				mapping_dict2["uni_pos"].append( line[7] )
				mapping_dict2["pdb_seq"].append( name_to_symbol( line[3] ) )
				mapping_dict2["pdb_pos"].append( line[4] )

	return mapping_dict1, mapping_dict2



# url = "https://www.ebi.ac.uk/pdbe/files/sifts/1a0n.xml"
# wget.download( url )

# exit()
################## SIFTS #################
##--------------------------------------##
def sifts_map_shell_command( pdb, max_trials = 10, wait_time = 5 ):
	"""
	Use SIFTS to download residue level PDB to Uniprot mapping.
	Using a Python library to directly fetch the mapping is slow.
		So, create a shell script to run the shell command on the go.
	Load the file and return the mapping.
		If invalid file saved, reattempt downloading until max_trials.

	Input:
	----------
	pdb --> PDB ID to be mapped.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	success --> (bool) True if successfully obtained the mapping.
	mapping --> PDB-Uiprot mapping in tsv format.
	"""

	script_name = f"map_pdb_to_uniprot_{pdb}.sh"
	pdb = pdb.lower()
	# Create a shell script to execute the curl command
	w = open( script_name, "w" )
	w.writelines( "#!/bin/bash \n" ) 
	w.writelines( "pdb=$1 \n" )
	w.writelines( "curl --silent https://www.ebi.ac.uk/pdbe/files/sifts/$pdb.xml.gz | gunzip | python parse_sifts.py > $pdb.tsv" )
	w.close()

	# print( "----------", pdb )
	# Make the shell script executable
	subprocess.call(["chmod", "+x", script_name])

	success = False

	for i in range( 1, max_trials ):			
		try:
			subprocess.call( ["sh", script_name, f"{pdb}"] )
			mapping = pd.read_csv( f"{pdb}.tsv", sep="\t" )
			success = True
			break

		# Some PDBs cannot be mapped to UniProt and no output is returned.
		except:
			# Wait for a few seconds.
			time.sleep( wait_time )

			if i == max_trials-1:
				mapping = 0
				break

	# if os.path.exists( f"{script_name}" ):
	subprocess.call( ["rm", f"{script_name}"] )
	
	return success, mapping

# sifts_map_shell_command( "1c4u", 5 )
# exit()

def download_SIFTS_Uni_PDB_mapping( mapped_PDB_path, pdb, max_trials = 10, wait_time = 5 ):
	"""
	Obtain PDB to Uniprot mapping using SIFTS if it doesn't already exist on disk.
	Success if the mapping exists and the .tsv file is readable.

	Input:
	----------
	mapped_PDB_path --> path for the mapping file.
	pdb --> PDB ID to be mapped.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	True if mapping already exists or has been successfully downloaded.
	"""
	if os.path.exists( f"{mapped_PDB_path}{pdb}.tsv" ):
		return True
	
	else:
		success, _ = sifts_map_shell_command( pdb, max_trials = max_trials, wait_time = wait_time )
		if success:
			return True
		
		else:
			return False



def get_sifts_mapping( mapping, chain1, uni_id1, chain2 = None, uni_id2 = None ):
	"""
	Get SIFTS PDB to Uniprot mapping for specified chains.
	SIFTS mapping contains the following info in order:
		PDB_ID	PDB_chain	PDB_res	PDB_pos 	Uniprot_ID 	Uniprot_res 	Uniprot_pos
	Chain ID in SIFTS mapping are Auth Asym IDs.
		For cif files, these could numeric or alphanumeric (e.g. 1c4u).
	Obtain all residues in mapping for the specified chain for 
		all Uniprot IDs associated with the chain.

	Input:
	----------
	mapping --> SIFTS mapping is tsv format.
	chain1 --> chain ID for which to obtain mapping.
	uni_id1 --> list of Uniprot IDs for chain1.
	chain2 --> chain ID for which to obtain mapping.
		If chain2 == None, only get chain1 mapping.
	uni_id2 --> list of Uniprot IDs for chain2.
		If uni_id2 == None, only get chain1 mapping.

	Returns:
	----------
	mapping_dict1 --> dict to store mapping info for chain1.
	mapping_dict2 --> dict to store mapping info for chain2.
	"""
	mapping_dict1 = {i:[] for i in ["pdb_seq", "pdb_pos", "uni_seq", "uni_pos"]}
	mapping_dict2 = {i:[] for i in ["pdb_seq", "pdb_pos", "uni_seq", "uni_pos"]}
	
	# Each chain ID should be mapped to a single Uniprot ID only.
	chain_1 = mapping[mapping.iloc[:,1].astype( str ) == chain1]
	chain_1 = chain_1[chain_1.iloc[:,4].isin( uni_id1 )]

	if len( chain_1 ) != 0:
		mapping_dict1["uni_id"] = str(chain_1.iloc[0,4])
		mapping_dict1["uni_seq"] = "".join( chain_1.iloc[:,5] )
		mapping_dict1["uni_pos"] = list( map( int, chain_1.iloc[:,6] ) )
		mapping_dict1["pdb_seq"] = "".join( chain_1.iloc[:,2] )
		mapping_dict1["pdb_pos"] = list( chain_1.iloc[:,3] )

	if chain2 != None:
		chain_2 = mapping[ mapping.iloc[:,1] == chain2 ]
		chain_1 = chain_1[chain_1.iloc[:,4].isin( uni_id2 )]
		# if len( pd.unique(chain_2.iloc[:,4] ) ) == 1:
		if len( chain_2 ) != 0:
			mapping_dict2["uni_id"] = str(chain_2.iloc[0,4])
			mapping_dict2["uni_seq"] = "".join( chain_2.iloc[:,5] )
			mapping_dict2["uni_pos"] = list( map( int, chain_2.iloc[:,6] ) )
			mapping_dict2["pdb_seq"] = "".join( chain_2.iloc[:,2] )
			mapping_dict2["pdb_pos"] = list( chain_2.iloc[:,3] )
	# mapping_dict2 will be empty if chain2 = None.
	return mapping_dict1, mapping_dict2

# mapping = pd.read_csv( "../Database/Mapped_PDBs/1dug.tsv", sep = "\t", header = None )
# m1, _ = get_sifts_mapping( mapping, chain1 = "A", uni_id1 = "P08515", chain2 = None, uni_id2 = None )
# print( m1 )
# exit()


################ PDB REST ################
##--------------------------------------##
def get_pdb_entry_info( entry_id, max_trials = 10, wait_time = 5 ):
	"""
	Retrieve PDB Entry level info from PDB REST API.

	Input:
	----------
	entry_id --> PDB ID.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	entry_data --> dict containing entry level info for the specified entry_id.
	"""
	entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
	entry_data = send_request( entry_url, _format = "json", 
								max_trials = max_trials, 
								wait_time = wait_time )

	return entry_data

# print( get_pdb_entry_info( "6vu3" ) )
# exit()


def get_pdb_entity_info( entry_id, entity_id, max_trials = 10, wait_time = 5 ):
	"""
	Retrieve PDB Entity level info from PDB REST API for the entry_id and entity_id.

	Input:
	----------
	entry_id --> PDB ID.
	entity_id --> Entity ID for associated with a PDB entry.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	entity_data --> dict containing entity level info for the specified entry_id and entry_id.
	"""
	entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
	entity_data = send_request( entity_url, _format = "json", 
								max_trials = max_trials, 
								wait_time = wait_time )

	return entity_data


def get_pdb_instance_info( entry_id, entity_id, asym_id, max_trials = 10, wait_time = 5 ):
	"""
	Retrieve PDB Entity level info from PDB REST API for the entry_id, entity_id, and asym_id.

	Input:
	----------
	entry_id --> PDB ID.
	entity_id --> Entity ID for associated with a PDB entry.
	asym_id --> Asym ID associated with an entity of a PDB entry.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	instance_data --> dict containing entity level info for the specified entry_id, entry_id, and asym_id.
	"""
	print( entry_id, "\t", entity_id, "\t", asym_id )
	instance_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{entry_id}/{asym_id}"
	instance_data = send_request( instance_url, _format = "json", 
								max_trials = max_trials, 
								wait_time = wait_time )

	return instance_data


def codename_Protein( entry_data ):
	"""
	Find out if the PDB contains only proteins or not.
		rcsb_entry_info
			polymer_composition
			polymer_entity_count
			polymer_entity_count_protein

	Input:
	----------
	entry_data --> PDB Entity level info obtained from PDB REST API.

	Returns:
	----------
	all_protein --> (bool) True ifonly protein entity present in PDB else False.
	"""
	all_protein = False
	nonpolymer_entity = entry_data["rcsb_entry_info"]["nonpolymer_entity_count"]
	polymer_entity_count = entry_data["rcsb_entry_info"]["polymer_entity_count"]
	count_protein = entry_data["rcsb_entry_info"]["polymer_entity_count_protein"]
	count_dna = entry_data["rcsb_entry_info"]["polymer_entity_count_dna"]
	count_rna = entry_data["rcsb_entry_info"]["polymer_entity_count_rna"]

	# print( nonpolymer_entity, "\t", polymer_entity_count, "\t",count_protein, "\t", count_dna, "\t", count_rna )
	if polymer_entity_count == count_protein:
		all_protein = True
	# For Sanity's sake.
	elif count_dna != 0 or count_rna != 0:
		all_protein = False
	elif nonpolymer_entity != 0:
		all_protein = False
	

	return all_protein

# entry_data =  get_pdb_entry_info( "1cqt" ) # 1rxv
# print( codename_Protein( entry_data ) )
# exit()

def is_chimera( uni_ids ):
	"""
	Chimeric proteins will have >1 Uniprot IDs,
		each belonging to a distinct protein.
	Check if all Uniprot IDs are the same or not.
		All should have the same sequence.

	Input:
	----------
	uni_ids --> list of Uniprot IDs.

	Returns:
	----------
	chimera --> (bool) True any of the protein sequences do not match.
	"""
	sequences = [get_uniprot_seq( id_ ) for id_ in uni_ids]
	sequences = [seq for seq in sequences if seq != []]
	
	chimera = any( [seq != sequences[0] for seq in sequences] )
	# print( chimera )
	# exit()

	return chimera



def from_pdb_rest_api_with_love( entry_id, max_trials = 10, wait_time = 5, custom = [None, None] ):
	"""
	Obtain Chain and UniProt IDs using the PDB ID.

	Input:
	----------
	entry_id --> PDB ID.
	max_trials --> maximum no. of attempts to fetch info. from the URL.
	wait_time --> waiting time before sending a request to the server again.
	custom --> If not None, will use the PDB Entry and Entity data dicts provided.

	Returns:
	----------
	0 if error obtaining info from REST API 
	None if entity information or entry information does not exist
	tuple of 
	- dataframe 
	- entry_data
	- entity_data 
	- list of entities with chimeric chains
	- list of entities with non-protein chains 

	"""
	df = pd.DataFrame( columns = ["PDB ID", "Entity ID", "Asym ID", "Auth Asym ID", "Uniprot ID"] )
	np_entity, chimeric, total_chains = [], [], 0
	all_uni_ids = []

	entry_data, entity_data = custom
	# If pre-existing data file does not exist.
	if entry_data == None:  # TODO there is no else statement to use pre-existing data file?
		entry_data = get_pdb_entry_info( entry_id, max_trials = max_trials, wait_time = wait_time )

	# PDB ID does not exist.
	if entry_data == "not_found": # TODO this will depend on previous condition?
		return None
 
	else:
		# Initialize an empty dataframe to store all relevant info for the PDB ID.
		# Obtain the entity ids in the PDB.
		all_entity_ids = entry_data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]

		entity_ids = []
		asym_ids, auth_asym_ids, uniprot_ids = {}, {}, {}
		
		# Dict to store entity data from PDB API for each entity ID.
		# Use existing one else create anew.
		entity_dict = {} if entity_data == None else entity_data  # TODO this can be taken inside the for loop. Redundant.

		row_idx = 0

		# All the IDs are strings.
		# For all the entities in the PDB.
		for entity_id in all_entity_ids:
			# If using pre-downloaded data, do not download again.
			if entity_data == None:
				entity_dict[entity_id] = get_pdb_entity_info( entry_id, entity_id, max_trials = max_trials, wait_time = wait_time )

				if entity_dict[entity_id] == "not_found":
					return None

			# Count all chains that exist in the PDB - protein and non-protein.
			total_chains += len( entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["asym_ids"] )

			if "uniprot_ids" in entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"].keys():
				asym_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["asym_ids"]
				auth_asym_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
				uniprot_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"]

				all_uni_ids.extend( uniprot_ids )

				# Remove chimeric entities.
				if len( uniprot_ids ) > 1:
					chimera = is_chimera( uniprot_ids )
				else:
					chimera = False

				if not chimera:

					for i in range( len( asym_ids ) ):
						df.loc[row_idx] = [
								entry_id, 
								entity_id,
								asym_ids[i],
								auth_asym_ids[i],
								",".join( uniprot_ids )
									]
						row_idx += 1
				else:
					chimeric.append( f"{entry_id}_{entity_id}" )

			else:
				np_entity.append( f"{entry_id}_{entity_id}" )

		return df, entry_data, entity_dict, chimeric, np_entity, total_chains, all_uni_ids
		# return [entity_ids, asym_ids, auth_asym_ids, uniprot_ids, [entry_data, entity_dict]]


# # JSON decoder error - 7z1n
# Chimeric 7pv1 (micos), 7pv0 (micos)
# df, entry_data, entity_dict, chimeric, np_entity, total_chains, all_uni_ids = from_pdb_rest_api_with_love( "9C82" )

# print( x ) # 2bug, 4njz, 2kft, 4bld, 2P4A, 2kwn, 7z1n, 6f38, 4z5t, 5nzu, 4mgb, 7lbm, 6mzm, 7nkx, 2kft, 7xox
# exit()


def get_superseding_pdb_id( pdb_id, max_trials = 10, wait_time = 5 ):
	"""
	Obtain PDB IDs that supersedes the input PDB ID.

	Input:
	----------
	entry_id --> PDB ID.
	max_trials --> maximum no. of attempts to fetch info. from the URL.
	wait_time --> waiting time before sending a request to the server again.

	Returns:
	----------
	new_pdb_id --> superseeding PDB ID.
	"""
	url = f"https://data.rcsb.org/rest/v1/holdings/removed/{pdb_id}"
	data = send_request( url, 
						_format = "json", max_trials = 10, wait_time = wait_time  )
	
	# PDB ID is still active. No superseeding PDB ID exists.
	if data == "not_found":
		new_pdb_id = pdb_id
		# break

	else:
		if "id_codes_replaced_by" in data["rcsb_repository_holdings_removed"].keys():
			new_pdb_id = data["rcsb_repository_holdings_removed"]["id_codes_replaced_by"][0]
		# If PDB ID has become obsolete and no new ID has been assigned (e.g. 8fg2).
		else:
			new_pdb_id = None

	return new_pdb_id

# print( get_superseding_pdb_id( "8fg2" ) )
# exit()


########################################################################################
"""
Functions below this have been deprecated.
"""
########################################################################################
def map_chain_ids( entry_id, auth_chain_id ):
	# Convert the auth_asym_ids to asym_ids.
	# entry_id --> PDB ID.
	# chain_id --> auth_asym_id (specified by the authors in the PDB file).

	# URL to obtain Entry specific info.
	entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
	data = send_request( entry_url )

	# Obtain the entity ids in the PDB.
	entity_ids = data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]

	chain_ids, uniprot_ids = {}, {}
	# All the IDs are strings.
	# For all the entities in the PDB.
	for entity_id in entity_ids:
		# URL to obtain Polymer Entity specific info.
		entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
		data = send_request( entity_url )

		auth_asym_ids = data["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
		asym_ids = data["rcsb_polymer_entity_container_identifiers"]["asym_ids"]

		if auth_chain_id in auth_asym_ids:
			idx = auth_asym_ids.index(auth_chain_id)
			new_chain_id = asym_ids[idx]

		return new_chain_id


# print(map_chain_ids("4bld", "A"))
# exit()


def mask_of_the_wraith( entry_id, entity_id, mis_uniprot_id ):
	# Obtain all Uniprot annotations from PDB for an entity.
	# entry_id --> PDB ID.
	# entity_id --> macromolecule id in the PDB.
	# mis_uniprot_id --> uniprot id for which alternate uniprot id is to be searched.

	uni_url = f"https://data.rcsb.org/rest/v1/core/uniprot/{entry_id}/{entity_id}"
	data = send_request( uni_url )

	alt_uni_id = [id_ for id_ in data[0]["rcsb_uniprot_accession"] if mis_uniprot_id == id_]
	return alt_uni_id
# entry_id = "3ik5"
# entity_id = 1
# mis_uniprot_id = "Q5QGG3"
# mask_of_the_wraith( entry_id, entity_id, mis_uniprot_id )


############## PDB GraphQL ###############
##--------------------------------------##
def get_alignments_from_pdb( instance ):
	# Get the alignments from PDB GraphQL API.
	# instance --> should be given as PDB_ID.instance_id eg: 1MV0.A.
		# instance_id should be the asym_id.
	
	# Create the GraphQL query.
	field0 = "{"
	field1 = f"""
	  alignment(
	    from:PDB_INSTANCE,
	    to:UNIPROT,
	    queryId: "{instance}" 
	  )"""

	field2 = """{
	    query_sequence
	    target_alignment {
	      target_id
	      target_sequence
	      coverage{
	        query_coverage
	        query_length
	        target_coverage
	        target_length
	      }
	      aligned_regions {
	        query_begin
	        query_end
	        target_begin
	        target_end
	      }
	    }
	  }
	}"""

	request_url = "https://1d-coordinates.rcsb.org/graphql?query"

	graphql_query = f"{field0}{field1}{field2}"	

	data = {"query": graphql_query}
	json_data = json.dumps(data)

	response = requests.post(url=request_url, data=json_data)

	data = json.loads(response.text)
	align_dict = {}

	if data["data"]["alignment"]["target_alignment"] == None:
		print("No alignment info exists for this entry... \n")
		return align_dict
	
	# Obtain the relevent info (seq and pos for PDB and Uniprot entry).
	query_beg = data["data"]["alignment"]["target_alignment"][0]["aligned_regions"][0]["query_begin"]
	query_end = data["data"]["alignment"]["target_alignment"][0]["aligned_regions"][0]["query_end"]
	target_beg = data["data"]["alignment"]["target_alignment"][0]["aligned_regions"][0]["target_begin"]
	target_end = data["data"]["alignment"]["target_alignment"][0]["aligned_regions"][0]["target_end"]

	# align_dict["pdb_pos"] = np.arange(query_beg, query_end+1, 1)
	align_dict["uni_pos"] = np.arange(target_beg, target_end+1, 1)
	align_dict["pdb_seq"] = data["data"]["alignment"]["query_sequence"][query_beg:query_end]
	align_dict["uni_seq"] = data["data"]["alignment"]["target_alignment"][0]["target_sequence"][target_beg:target_end]

	return align_dict


# align_dict = get_alignments_from_pdb("1MV0.A")

# print(align_dict["pdb_pos"])
# print(align_dict["uni_pos"])



############ FuzzDB REST API #############
##--------------------------------------##
def fuzzdb_api( uniprot_acc, pdb ):
	# Obtain Chain IDs for FuzzDB entries using uniprot ID.
	# uniprot_acc --> Uniprot ID for the entry.
	
	# URL to obtain Entry specific info.
	entry_url = url = f"https://fuzdb.org/api/entries?uniprot_acc={uniprot_acc}"
	data = send_request( entry_url )

	chain_ids = []
	for i in range( len( data ) ):
		# for key1 in data[i].keys():
			# print( key1, " ------------------" )
			# print( data[i][key1] )
		break
		if "pdb_reference" in data[i].keys():
			for j in range( len( data[i]["pdb_reference"] ) ):
				for k in range( len( data[i]["pdb_reference"][j] ) ):
					chains = data[i]["pdb_reference"][j]["chains"]
					# Obtain chains for the specified PDB ID if the chains are present in FuzzDB API.
					if pdb.lower() in data[i]["pdb_reference"][j]["id"] and chains != []:
						# Multiple elements present with same details.
						chain_ids = [_id["id"] for _id in chains]
	if chain_ids != []:
		return chain_ids
	else:
		return None

# print( fuzzdb_api( "P13569", "" ) ) # P03069, 2lpb
# exit()

############ DisProt REST API ############
##--------------------------------------##
def disprot_api( id_, pdb = False ):
	# Obtain disordered residue positions using the uniprot id.
	# uniprot_acc --> Uniprot ID.
	
	# URL to obtain Entry specific info.
	if pdb:
		entry_url = url = f"https://disprot.org/api/search?disprot_id={id_}"
	else:
		entry_url = url = f"https://disprot.org/api/search?acc={id_}"
	data = send_request( entry_url )

	if data["data"] == []:
		name = "Disprot" if pdb else "Uniprot"
		print(f"No Disprot entry does not exist for {name} ID: {id_}... \n")
		return None
	else:
		if pdb:
			pdb_ids = []
			# For all the regions present in the entry.
			for i in range( len( data["data"][0]["regions"] ) ):
				if "cross_refs" in data["data"][0]["regions"][i].keys():
					# Obtain all PDB ids in the entry.
					[pdb_ids.append( x["id"] ) for x in data["data"][0]["regions"][i]["cross_refs"]]

			return list( set( pdb_ids ) )
		else:
			disprot_pos = []
			for i in range( len( data["data"][0]["regions"] ) ):
				start = data["data"][0]["regions"][i]["start"] 
				end = data["data"][0]["regions"][i]["end"]
				disprot_pos.append( np.arange( start, end+1, 1 ) )

			return disprot_pos

# print( disprot_api("P03069") )

"""
【中文解析-可演化改动建议】
1) 将硬编码路径迁移到统一配置文件（YAML/ENV）并加入路径合法性校验。
2) 为每个中间文件增加 schema 校验（字段名/类型/范围），降低跨脚本耦合导致的隐式错误。
3) 增加小样本 smoke test（例如 3~5 条记录）和数据快照测试，保障重构时行为一致。
4) 若面向预训练，建议输出 token 级别元信息（来源数据库、结构置信度、对齐覆盖率）便于构建多任务标签。
"""
