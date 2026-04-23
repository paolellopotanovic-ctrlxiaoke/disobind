"""
【中文解析-模块总览】
- 中心功能：get_ped_ensembles.py 属于 Disobind 数据流水线脚本，用于数据抓取/清洗/映射/导出中的一个环节。
- 逻辑位置：该文件在“原始数据库 -> 统一样本定义 -> 训练/评估输入文件”链条中承担局部处理任务。
- 输入输出流水线：通常输入为数据库文件（csv/tsv/xlsx/json）、结构文件（pdb/cif）与序列标识（UniProt/PDB），输出为中间表、映射字典、模型可读输入（csv/h5/json/npy/fasta）。
- 可选项：多数脚本支持通过构造参数或命令行参数控制版本号、并行核数、路径和过滤阈值。
- 可视化能力：数据脚本本身通常不直接出图；如有图像分析一般在 analysis/ 目录统一完成。
- 数据格式与实验标签：核心监督标签通常是残基-残基接触图（contact map）或界面位点（interface residues）；本目录负责把外部异构数据规范化成论文实验使用的统一格式。
- 数据来源与论文逻辑：数据来源包含 DIBS/MFIB/DisProt/IDEAL/MobiDB/PDB/SIFTS 等；处理逻辑服务于“按二元复合体构建样本并对齐序列-结构关系”。
- 预训练扩展建议：若做进一步预训练，建议先扩展原始来源覆盖面、统一长度分桶策略、保留更多中间质控字段（置信度/覆盖率/冲突标记），再生成更大规模弱监督样本。
"""

"""
Obtain entries from PEDS containing a binary complex.
Create input and output files for running Disobind.
"""
############ ------>"May the Force serve u well..." <------##############
#########################################################################
import json, os, subprocess, json
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import h5py
from Bio.PDB import MMCIFParser

from from_APIs_with_love import ( send_request, from_pdb_rest_api_with_love,
                                    download_SIFTS_Uni_PDB_mapping, get_sifts_mapping,
                                    download_pdb, get_uniprot_seq )
from utility import ( load_PDB, get_coordinates, get_contact_map )


# 【中文解析-关键函数/类入口】以下对象是本文件的主执行入口，承担数据转换主链路。
class APedsTale():
    """
    Download entries from PEDS database for protein-protein complexes
        along with other accessory information.
    Get input sequence pairs and corresponding output contact maps.
    """
    def __init__(self):
        self.base_dir = "../database/"
        self.peds_dir = os.path.join( self.base_dir, "PEDS/" )
        self.peds_data_dir = os.path.join( self.peds_dir, "data/" )
        self.peds_ensemble_dir = os.path.join( self.peds_dir, "ensemble/" )
        self.peds_pdb_details_dir = os.path.join( self.peds_dir, "pdb_api/" )
        self.peds_sifts_dir = os.path.join( self.peds_dir, "ped_sifts/" )
        self.peds_pdb_struct_dir = os.path.join( self.peds_dir, "ped_pdb_struct/" )
        self.af2_fasta_dir = os.path.join( self.peds_dir, "AF2_fasta_peds/" )
        self.af3_json_dir = os.path.join( self.peds_dir, "AF3_json_peds/" )

        self.valid_ped_entries_file = os.path.join( self.peds_dir, "valid_ped_entries.txt" )
        self.ped_pdbs_file = os.path.join( self.peds_dir, "ped_pdb_ids.txt" )
        self.peds_uni_seq_file = os.path.join( self.peds_dir, "Uniprot_seq_PEDS.json" )
        self.peds_test_input = os.path.join( self.peds_dir, "ped_test_input.csv" )
        self.peds_test_target = os.path.join( self.peds_dir, "ped_test_target.h5" )
        self.af2_input_file = os.path.join( self.peds_dir, "AF2_peds_fasta_paths.txt" )
        
        if os.path.exists( self.peds_uni_seq_file ):
            with open( self.peds_uni_seq_file, "r" ) as f:
                self.uni_seq_dict = json.load( f )
        else:
            self.uni_seq_dict = {}

        self.max_len = 200
        self.contact_threshold = 8
    # 【中文解析-重点逻辑】forward 通常串联“读取输入 -> 清洗/映射 -> 导出中间/最终文件”，是该脚本最关键的数据流水线节点。
    def forward( self ):
        self.create_dir()
        peds_dict, peds_data = self.get_ped_entries()
        self.create_af2_input( peds_data )
        self.create_af3_input( peds_data )
        self.save( peds_dict, peds_data )


    def create_dir( self ):
        """
        Create the required directories if they don't already exist.
        """
        os.makedirs( self.peds_dir, exist_ok = True )
        os.makedirs( self.peds_data_dir, exist_ok = True )
        os.makedirs( self.peds_ensemble_dir, exist_ok = True )
        os.makedirs( self.peds_pdb_details_dir, exist_ok = True )
        os.makedirs( self.peds_sifts_dir, exist_ok = True )
        os.makedirs( self.peds_pdb_struct_dir, exist_ok = True )
        os.makedirs( self.af2_fasta_dir, exist_ok = True )
        os.makedirs( self.af3_json_dir, exist_ok = True )


    def create_ped_ids( self ) -> List:
        """
        Assuming PEDS IDs are sequential starting from 1.
        530 = last PED entry ID + 1.
        """
        ped_ids = []
        for i in range( 1,530 ):
            if i<10:
                ped_ids.append( "PED0000"+str( i ) )
            elif i<100:
                ped_ids.append( "PED000"+str( i ) )
            else:
                ped_ids.append( "PED00"+str( i ) )
        return ped_ids


    def get_ped_url( self, num_id: int ) -> str:
        base_url = f"https://deposition.proteinensemble.org/api/v1/entries/"
        ped_url = f"{base_url}{num_id}"
        return ped_url


    def entry_is_valid( self, data: Dict ) -> bool:
        """
        We assume a PED ID to be invalid if the data couldn't be
            retrieved even after max_trials.
        """
        # Ignore if the entry_id does not exist.
        if data in ["not_found", "bad_request"]:
            valid = False            
        # Save the existent PED IDs.
        else:
            valid = True
        return valid


    def is_binary_complex( self, data: Dict ) -> bool:
        """
        "construct_chains" -> list[Dict for all chains]
        For complexes, the "construct_chains" key should have length >1.
        """
        if len( data["construct_chains"] ) != 2:
            is_binary_complex = False
        # PED entries containing complexes.
        else:
            is_binary_complex = True
        return is_binary_complex


    def has_missing_residues( self, data: Dict ) -> bool:
        """
        "construct_chains" -> list[Dict for all chains]
        "missing" key contains a list of Dict for all missing residues.
        True if any chain has missing residues.
        """
        missing = []
        for chain in data["construct_chains"]:
            if len( chain["missings"] ):
                # print( chain["missings"] )
                missing.append( True )
            else:
                missing.append( False )
        return any( missing )


    def check_seq_cmap_size( self, chain_uni_map: Dict, contact_map: np.array ):
        """
        Check if the lengths of the proteins match the size of contact maps or not.
        Asuuming a binary complex.
        """
        chain1, chain2 = chain_uni_map.keys()
        length1 = chain_uni_map[chain1]["length"]
        length2 = chain_uni_map[chain2]["length"]

        print( contact_map.shape, "  ", ( length1, length2 ), " --> ", contact_map.shape == ( length1, length2 ) )
        if contact_map.shape == ( length1, length2 ):
            success = True
        else:
            success = False

        return success


    def get_ped_ids( self ) -> List:
        """
        Load valid PED IDs or create anew.
        """
        if os.path.exists( self.valid_ped_entries_file ):
            with open( self.valid_ped_entries_file, "r" ) as f:
                ped_ids = f.readlines()[0].split( "," )[:-1]
        else:
            ped_ids = self.create_ped_ids()
        return ped_ids


    def get_data_from_peds_api( self, num_id: int ) -> Tuple[Dict, bool]:
        """
        Get the data from PEDS and save as a JSON dict.
        """
        json_path = os.path.join( self.peds_data_dir, f"{num_id}.json" )
        ped_url = self.get_ped_url( num_id )
        
        if os.path.exists( json_path ):
            with open( json_path, "r" ) as f:
                data = json.load( f )
        else:
            data = send_request( ped_url, _format = "json", max_trials = 10, wait_time = 5 )

        entry_valid = self.entry_is_valid( data )
        if entry_valid:
            # Save the JSON file for the entry.
            with open( json_path, "w" ) as w:
                json.dump( data, w )
        return data, entry_valid


    def get_pdb_ids_from_entry( self, data: Dict ) -> str:
        """
        Given the PEDS entry dict, get the PDB ID for the entry.
        "entry_cross_reference" key is always present.
            If no cross-ref is available, it's an empty list.
        """
        if data["description"]["entry_cross_reference"] != []:
            for cref in data["description"]["entry_cross_reference"]:
                if cref["db"] == "pdb":
                    pdb_id = cref["id"].lower()
                else:
                    pdb_id = ""
        else:
            pdb_id = ""
        return pdb_id


    def get_data_from_pdb_api( self, pdb_id: str ) -> pd.DataFrame:
        """
        Get the entry and entity details for a
            given PDB ID using the PDB REST API.
        This considers only protein chains.
        """
        file_path = os.path.join( self.peds_pdb_details_dir, f"{pdb_id}.csv" )
        if not os.path.exists( file_path ):
            result = from_pdb_rest_api_with_love( pdb_id )
            # df, entry_data, entity_dict, chimeric, np_entity, total_chains, all_uni_ids = result
            df, _, _, _, _, _, _ = result
            
            df.to_csv( file_path, index = False )
        else:
            df = pd.read_csv( file_path )

        return df


    def get_chain_uniprot_mapping( self, df: pd.DataFrame ) -> Dict:
        """
        Map the UniProt ID to the corresponding Auth Asym ID.
        """
        chain_uni_map = {}
        for i in range( df.shape[0] ):
            auth_asym_id = df.loc[i, "Auth Asym ID"]
            uni_id = df.loc[i, "Uniprot ID"]

            for aa_id in auth_asym_id.split( "," ):
                chain_uni_map[aa_id] = {}
                chain_uni_map[aa_id]["uni_id"] = uni_id.split( "," )
        return chain_uni_map


    def get_pdb_uni_mapping( self, pdb_id: str ) -> Tuple[bool, pd.DataFrame]:
        """
        Map PDb to UniProt using SIFTS.
        """
        print( f"SIFTS mapping for {pdb_id}..." )
        sifts_file_path = os.path.join( self.peds_sifts_dir, f"{pdb_id}.tsv" )
        if os.path.exists( sifts_file_path  ):
            success = True
        else:
            success = download_SIFTS_Uni_PDB_mapping( self.peds_sifts_dir,
                                                        pdb_id,
                                                        max_trials = 10, wait_time = 5 )
            subprocess.call( ["mv", f"./{pdb_id}.tsv", sifts_file_path] )
        
        if success:
            mapping = pd.read_csv( sifts_file_path, sep = "\t", header = None )
        else:
            mapping = pd.DataFrame( {} )
        return success, mapping


    def get_uniprot_feats( self, mapping: pd.DataFrame, chain_uni_map: Dict ) -> Dict:
        """
        Get UniProt residue positions from the SIFTS mapping for each chain.
        """
        max_len_exceed = []
        for chain in chain_uni_map:
            uni_id = chain_uni_map[chain]["uni_id"][0]
            mapping_dict, _ = get_sifts_mapping( mapping = mapping, chain1 = chain, uni_id1 = [uni_id] )

            if uni_id != mapping_dict["uni_id"]:
                raise ValueError( "Mismatch in PDB and SIFTS UniProt ID..." )

            if len( mapping_dict["uni_pos"] ) > self.max_len:
                max_len_exceed.append( True )

            uni_start_pos = mapping_dict["uni_pos"][0]
            uni_end_pos = mapping_dict["uni_pos"][-1]
            chain_uni_map[chain]["pdb_pos"] = mapping_dict["pdb_pos"]
            chain_uni_map[chain]["uni_res"] = [uni_start_pos, uni_end_pos]
            chain_uni_map[chain]["length"] = uni_end_pos - uni_start_pos + 1

        return chain_uni_map, any( max_len_exceed )


    def download_pdb_struct( self, pdb_id: str ) -> bool:
        """
        Given the PDB ID download the structure file in .cif format.
        """
        ext, _ = download_pdb( pdb_id, max_trials = 5, wait_time = 5, return_id = True )

        if ext == None:
            success = False
        else:
            pdb_file_path = os.path.join( self.peds_pdb_struct_dir, f"{pdb_id}.{ext}" )
            subprocess.call( ["mv", f"./{pdb_id}.{ext}", pdb_file_path] )
            success = True

        return success


    def download_uniprot_seq( self, chain_uni_map: Dict ):
        """
        Download UniProt seq for all UniProt IDs in the input dict.
        """
        success = []
        for chain in chain_uni_map:
            uni_id = chain_uni_map[chain]["uni_id"][0]
            
            if uni_id in self.uni_seq_dict:
                success.append( True )
            else:
                uni_seq = get_uniprot_seq( uni_id, max_trials = 5,
                                            wait_time = 5,
                                            return_id = False )
                if uni_seq != []:
                    success.append( True )
                    self.uni_seq_dict[uni_id] = uni_seq
                else:
                    success.append( False )
        return all( success )


    def get_coordinates_from_pdb( self, pdb_id: str, chain_uni_map:Dict ) -> Dict:
        """
        A generator object that yields a dict containing
            coordinates for all chains in a model.
        """
        structure = load_PDB( pdb = pdb_id,
                            pdb_path = self.peds_pdb_struct_dir )
        
        coords_dict = {}
        for model in structure:
            for chain in model:
                chain_id = chain.id[0]
                if chain_id in chain_uni_map:
                    res_pos = chain_uni_map[chain_id]["pdb_pos"]
                    coords_dict[chain_id] = get_coordinates( chain, res_pos )
            yield coords_dict


    def create_contact_maps( self, pdb_id: str, chain_uni_map:Dict ) -> np.array:
        """
        Get the coordinates for all models in a PDB structure and create contact maps.
        Create summed contact maps and convert to binary contact maps.
        """
        print( f"Creating contact map for {pdb_id}..." )
        contact_map  =np.array( [] )
        for coords_dict in self.get_coordinates_from_pdb( pdb_id, chain_uni_map ):
            coords1, coords2 = coords_dict.values()
            if contact_map.shape[0] == 0:
                contact_map = get_contact_map( coords1, coords2, self.contact_threshold )
            else:
                contact_map += get_contact_map( coords1, coords2, self.contact_threshold )
        contact_map = np.where( contact_map > 0, 1, 0 )
        return contact_map


    def create_uni_id_pairs( self, chain_uni_map: Dict ):
        """
        Create UniProt ID pairs in the same format as the training and OOD sets:
            "{Uni_ID1}:start_res:end_res--{Uni_ID2}:start_res:end_res_{copy_num}"
        Assuimg a binary complex.
        """
        chain1, chain2 = chain_uni_map.keys()
        uni_id1 = chain_uni_map[chain1]["uni_id"][0]
        res1 = chain_uni_map[chain1]["uni_res"]
        uni_id2 = chain_uni_map[chain2]["uni_id"][0]
        res2 = chain_uni_map[chain2]["uni_res"]

        uni_id_pair = f"{uni_id1}:{res1[0]}:{res1[1]}--{uni_id2}:{res2[0]}:{res2[1]}_0"

        print( uni_id_pair )
        return uni_id_pair


    def get_prot_seq( self, entry_id: str ):
        """
        Get the required prot1/2 seq for the given entry.
        """
        uni_id1, uni_id2 = entry_id.split( "--" )
        uni_id2, _ = uni_id2.split( "_" )
        uni_id1, start1, end1 = uni_id1.split( ":" )
        uni_id2, start2, end2 = uni_id2.split( ":" )

        prot1_seq = self.uni_seq_dict[uni_id1][int( start1 )-1:int( end1 )]
        prot2_seq = self.uni_seq_dict[uni_id2][int( start2 )-1:int( end2 )]
        return prot1_seq, prot2_seq


    def get_ped_entries( self ):
        """
        Get data from PEDS API for all the valid PEDS IDs.
        Save the data as a JSON file.
        Select only entries containing complexes.
        Get PDB IDs for each.
        Get UniProt IDs using the PDB ID.
        Map PDB to UniProt using SIFTS.
        Get the UniProt start, end residue positions from SIFTS mapping.
        Remove if:
            Not a valid PEDS ID.
            Has missing residues.
            Not a binary complex entry.
            Has no PDB ID.
            PDB has non-protein chains too.
            No SIFTS mapping.
            Any chain in PDB exceed max length.
            Mismatch in seq length and cmap dim.
        """
        peds_dict = {k:[] for k in ["valid_ped_entries", "pdb_ids"]}
        peds_data = {"entry_id": [], "cmap": {}, "seq": {}, "pdb": {}}
        ped_ids = self.get_ped_ids()

        for i, num_id in enumerate( ped_ids ):
            # num_id = "PED00002"
            # num_id = "PED00105"
            print( "\n-------> ", num_id )

            data, entry_valid = self.get_data_from_peds_api( num_id )

            # print( data.keys() )
            # # print( data["construct_chains"][0].keys() )
            # # print( data["construct_chains"][0]["chain_name"] )
            # # print( data["construct_chains"][0]["alignment"] )
            # print( "\n" )
            # print( data["description"].keys() )
            # print( data["description"]["entry_cross_reference"] )
            # print( data["description"]["experimental_cross_reference"] )
            # print( data["ensembles"] )
            # exit()
            if not entry_valid:
                continue
            peds_dict["valid_ped_entries"].append( num_id )

            # Contains missing residues.
            missing = self.has_missing_residues( data )
            if missing:
                print( "Missing residues" )
                continue
            if not self.is_binary_complex( data ):
                continue

            print( f"Here exists a complex entry {num_id}..." )
            pdb_id = self.get_pdb_ids_from_entry( data )
            # No PDB entry available.
            if pdb_id == "":
                continue

            # if pdb_id in peds_dict["pdb_ids"]:
            #     continue

            # Redundant with training set and multiple insulin struct.
            # if pdb_id in ["2bn5", "2mh3", "5nwm", "6b7g"] + ["1jco", "2mvd", "2rn5"]:
            #     continue

            # Cherry-picked entries.
            # if pdb_id not in ["1hui", "2dt7"."2jwn", "2n9p", "5tmx", "2mkr", "2mps", "2n3a"]:
            if pdb_id not in ["2dt7", "2jwn", "2n3a"]:
                continue

            print( f"-----------------> {pdb_id}" )
            df = self.get_data_from_pdb_api( pdb_id )
            # Should have 2 or more protein chains.
            if df.shape[0] > 1:
                chain_uni_map = self.get_chain_uniprot_mapping( df )

                success, mapping = self.get_pdb_uni_mapping( pdb_id )
                if not success:
                    continue
                chain_uni_map, max_len_exceed = self.get_uniprot_feats( mapping, chain_uni_map )

                if max_len_exceed:
                    print( f"{num_id} -> {pdb_id} exceds max length..." )
                    continue

                struct_success = self.download_pdb_struct( pdb_id )
                if not struct_success:
                    continue

                seq_success = self.download_uniprot_seq( chain_uni_map )
                if not seq_success:
                    continue

                contact_map = self.create_contact_maps( pdb_id, chain_uni_map )

                success_size = self.check_seq_cmap_size( chain_uni_map, contact_map )
                if not success_size:
                    continue

                # if np.count_nonzero( contact_map ) == 0:
                #     print( f"No contacts present at 8Angstorm in PDB: {pdb_id}..." )
                #     continue
                uni_id_pair = self.create_uni_id_pairs( chain_uni_map )
                prot1_seq, prot2_seq = self.get_prot_seq( uni_id_pair )
                peds_data["entry_id"].append( uni_id_pair )
                peds_data["cmap"][uni_id_pair] = contact_map
                peds_data["seq"][uni_id_pair] = [prot1_seq, prot2_seq]
                peds_data["pdb"][uni_id_pair] = pdb_id
                peds_dict["pdb_ids"].append( pdb_id )

        print( "Total valid PED entries found: ", len( peds_dict["valid_ped_entries"] ) )
        print( "Total PDB IDs obtained from PEDS: ", len( peds_dict["pdb_ids"] ) )

        return peds_dict, peds_data


    def create_af2_input( self, peds_data:Dict ):
        """
        Create a directory containng fasta files to be used as input by AF2.
            Each file contains Uniprot seq for the protein pairs with 
                the respective Uniprot IDs as header.
            e.g. "{entry}.fasta" 
                >"{Uni_ID1}:start:end"
                Sequence for prot1

                >"{Uni_ID2}:start:end"
                Sequence for prot2
            start, end refer to the the first and last residue positions.
        Also dump the paths to all FASTA files in a comma-separated txt file 
            for ease of running AF2.
        """
        if not os.path.exists( self.af2_fasta_dir ): 
            os.makedirs( f"{self.af2_fasta_dir}" )

        w_path =  open( self.af2_input_file, "w" )
        with open( self.af2_input_file, "w" ) as w:
            for entry_id in peds_data["seq"].keys():
                prot1_seq, prot2_seq = peds_data["seq"][entry_id]

                uni_id1, uni_id2 = entry_id.split( "--" )
                uni_id2, copy_num = uni_id2.split( "_" )
                uni_id1, s1, e1 = uni_id1.split( ":" )
                uni_id2, s2, e2 = uni_id2.split( ":" )

                entry_id = f"{uni_id1}--{uni_id2}_{copy_num}"

                with open( f"{self.af2_fasta_dir}{entry_id}.fasta", "w" ) as w:
                    w.writelines( ">" + f"{uni_id1}:{s1}:{e1}" + "\n" + prot1_seq + "\n\n" )
                    w.writelines( ">" + f"{uni_id2}:{s2}:{e2}" + "\n" + prot2_seq + "\n" )

                w_path.writelines( f"./AF2_fasta_peds/{entry_id}.fasta," )
        w_path.close()


    def create_af3_input( self, peds_data: Dict ):
        """
        Create JSON file for batch running AF3 server.
        Will create batches of 20 merged binary complexes.
        AF3 requires a list of dict in JSON format 
                (https://github.com/google-deepmind/alphafold/blob/main/server/README.md).
        The entry_id serves as the job name.
        This allows to upload 20 jobs on the AF3 server at once,
                but you still need to run each job one by one.
        """
        peds_entry_ids = list( peds_data["seq"].keys() )
        for start in np.arange( 0, len( peds_entry_ids ), 30 ):
            if ( start + 30 ) > len( peds_entry_ids ):
                end = len( peds_entry_ids )
            else:
                end = start + 30
            af3_batch = []
            for entry_id in peds_entry_ids[start:end]:
                prot1_seq, prot2_seq = peds_data["seq"][entry_id]

                uni_id1, uni_id2 = entry_id.split( "--" )
                uni_id2, copy_num = uni_id2.split( "_" )
                uni_id1, s1, e1 = uni_id1.split( ":" )
                uni_id2, s2, e2 = uni_id2.split( ":" )

                entry_id = f"{uni_id1}--{uni_id2}_{copy_num}"

                af3_entry = {}
                af3_entry["name"] = entry_id
                af3_entry["modelSeeds"] = [1]
                af3_entry["sequences"] = [
                                    {
                                    "proteinChain": {
                                            "sequence": prot1_seq,
                                            "count": 1
                                    } },
                                    {
                                    "proteinChain": {
                                            "sequence": prot2_seq,
                                            "count": 1
                                    } }
                ]

                af3_batch.append( af3_entry )

            # Save batches of 20 merged binary complexes.
            with open( f"{self.af3_json_dir}Batch_{start}-{end}.json", "w" ) as w:
                json.dump( af3_batch, w )


    def save( self, peds_dict: Dict, peds_data: Dict ):
        # if not os.path.exists( self.valid_ped_entries_file ):
        with open( self.valid_ped_entries_file, "w" ) as w:
            w.writelines( ",".join( sorted( peds_dict["valid_ped_entries"] ) ) )

        # if not os.path.exists( self.ped_pdbs_file ):
        with open( self.ped_pdbs_file, "w" ) as w:
            w.writelines( ",".join( sorted( peds_dict["pdb_ids"] ) ) )

        with open( self.peds_uni_seq_file, "w" ) as w:
            json.dump( self.uni_seq_dict, w )

        with open( self.peds_test_input, "w" ) as w:
            w.writelines( ",".join( peds_data["entry_id"] ) )

        hf = h5py.File( self.peds_test_target, "w" )
        for k in peds_data["cmap"]:
            print( k, " --> ", peds_data["pdb"][k], "  ",
                    peds_data["cmap"][k].shape, "  ",
                    np.count_nonzero( peds_data["cmap"][k] ), "  ",
                    round( np.count_nonzero( peds_data["cmap"][k] )/peds_data["cmap"][k].size, 3 ) )
            hf.create_dataset( k, data = peds_data["cmap"][k] )


APedsTale().forward()

"""
【中文解析-可演化改动建议】
1) 将硬编码路径迁移到统一配置文件（YAML/ENV）并加入路径合法性校验。
2) 为每个中间文件增加 schema 校验（字段名/类型/范围），降低跨脚本耦合导致的隐式错误。
3) 增加小样本 smoke test（例如 3~5 条记录）和数据快照测试，保障重构时行为一致。
4) 若面向预训练，建议输出 token 级别元信息（来源数据库、结构置信度、对齐覆盖率）便于构建多任务标签。
"""
