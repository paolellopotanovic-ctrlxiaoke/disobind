"""
【中文解析-模块总览】
- 中心功能：prepare_entry_from_pdb.py 属于 Disobind 数据流水线脚本，用于数据抓取/清洗/映射/导出中的一个环节。
- 逻辑位置：该文件在“原始数据库 -> 统一样本定义 -> 训练/评估输入文件”链条中承担局部处理任务。
- 输入输出流水线：通常输入为数据库文件（csv/tsv/xlsx/json）、结构文件（pdb/cif）与序列标识（UniProt/PDB），输出为中间表、映射字典、模型可读输入（csv/h5/json/npy/fasta）。
- 可选项：多数脚本支持通过构造参数或命令行参数控制版本号、并行核数、路径和过滤阈值。
- 可视化能力：数据脚本本身通常不直接出图；如有图像分析一般在 analysis/ 目录统一完成。
- 数据格式与实验标签：核心监督标签通常是残基-残基接触图（contact map）或界面位点（interface residues）；本目录负责把外部异构数据规范化成论文实验使用的统一格式。
- 数据来源与论文逻辑：数据来源包含 DIBS/MFIB/DisProt/IDEAL/MobiDB/PDB/SIFTS 等；处理逻辑服务于“按二元复合体构建样本并对齐序列-结构关系”。
- 预训练扩展建议：若做进一步预训练，建议先扩展原始来源覆盖面、统一长度分桶策略、保留更多中间质控字段（置信度/覆盖率/冲突标记），再生成更大规模弱监督样本。
"""

"""
Create input and output files for running Disobind for given PDB IDs.
"""
############ ------>"May the Force serve u well..." <------##############
#########################################################################
import math
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
class EntryFromPDB():
    """
    Get input sequence pairs and corresponding output contact maps.
    """
    def __init__( self ):
        self.base_dir = "../database/"
        self.misc_dir = os.path.join( self.base_dir, "Misc/" )
        self.misc_data_dir = os.path.join( self.misc_dir, "data/" )
        self.misc_ensemble_dir = os.path.join( self.misc_dir, "ensemble/" )
        self.misc_pdb_details_dir = os.path.join( self.misc_dir, "pdb_api/" )
        self.misc_sifts_dir = os.path.join( self.misc_dir, "misc_sifts/" )
        self.misc_pdb_struct_dir = os.path.join( self.misc_dir, "misc_pdb_struct/" )
        self.af2_fasta_dir = os.path.join( self.misc_dir, "AF2_fasta_misc/" )
        self.af3_json_dir = os.path.join( self.misc_dir, "AF3_json_misc/" )

        self.misc_pdbs_file = os.path.join( self.misc_dir, "misc_pdb_ids.txt" )
        self.misc_uni_seq_file = os.path.join( self.misc_dir, "Uniprot_seq_misc.json" )
        self.misc_test_input = os.path.join( self.misc_dir, "misc_test_input.csv" )
        self.misc_test_target = os.path.join( self.misc_dir, "misc_test_target.h5" )
        self.af2_input_file = os.path.join( self.misc_dir, "AF2_misc_fasta_paths.txt" )
        self.complexes_summary_file = os.path.join( self.misc_dir, "Summary.json" )

        self.select_chains = {
            # "1dt7": [],
            "2lmo": ["A", "B"],
            "2lmp": ["A", "B"],
            "2lmq": ["A", "B"],
            "8cmk": ["A", "C"],
            "7lna": ["A", "B"],
            "6xmn": ["A", "B"]
            }
        self.select_res = {
            "8cmk": {"A": np.arange( 700, 801, 1 ),
                    "C": []},
            "7lna": {"A": np.arange( 95, 194, 1 ),
                    "B": np.arange( 95, 194, 1 )},
            # "6xmn": {"A": np.arange( 700, 800, 1 ),
            #         "C": []}
            }

        if os.path.exists( self.misc_uni_seq_file ):
            with open( self.misc_uni_seq_file, "r" ) as f:
                self.uni_seq_dict = json.load( f )
        else:
            self.uni_seq_dict = {}

        self.max_len = 200
        self.contact_threshold = 8
    # 【中文解析-重点逻辑】forward 通常串联“读取输入 -> 清洗/映射 -> 导出中间/最终文件”，是该脚本最关键的数据流水线节点。
    def forward( self ):
        self.create_dirs()
        misc_data = self.get_misc_entries()
        # Get AF2/3 input files.
        self.create_af2_input( misc_data )
        self.create_af3_input( misc_data )
        self.save( misc_data )


    def create_dirs( self ):
        """
        Create the required directories if they don't already exist.
        """
        os.makedirs( self.misc_dir, exist_ok = True )
        os.makedirs( self.misc_pdb_details_dir, exist_ok = True )
        os.makedirs( self.misc_sifts_dir, exist_ok = True )
        os.makedirs( self.misc_pdb_struct_dir, exist_ok = True )
        os.makedirs( self.af2_fasta_dir, exist_ok = True )
        os.makedirs( self.af3_json_dir, exist_ok = True )


    def check_seq_cmap_size( self, chain_uni_map: Dict, contact_map: np.array ):
        """
        Check if the lengths of the proteins match the size of contact maps or not.
        Asuuming a binary complex.
        """
        chain1, chain2 = chain_uni_map.keys()
        length1 = chain_uni_map[chain1]["length"]
        length2 = chain_uni_map[chain2]["length"]

        if contact_map.shape == ( length1, length2 ):
            success = True
        else:
            print( contact_map.shape, "  ", ( length1, length2 ) )
            success = False

        return success


    def too_many_chains( self, chain_uni_map: Dict ) -> bool:
        """
        Check if the PDB has >2 chains.
        """
        if len( chain_uni_map ) > 2:
            too_many = True
        else:
            too_many = False
        return too_many


    def get_data_from_pdb_api( self, pdb_id: str ) -> pd.DataFrame:
        """
        Get the entry and entity details for a
            given PDB ID using the PDB REST API.
        This considers only protein chains.
        """
        file_path = os.path.join( self.misc_pdb_details_dir, f"{pdb_id}.csv" )
        if not os.path.exists( file_path ):
            result = from_pdb_rest_api_with_love( pdb_id )
            # df, entry_data, entity_dict, chimeric, np_entity, total_chains, all_uni_ids = result
            df, _, _, _, _, _, _ = result
            
            df.to_csv( file_path, index = False )
        else:
            df = pd.read_csv( file_path )

        return df


    def get_chain_uniprot_mapping( self, pdb_id: str, df: pd.DataFrame ) -> Dict:
        """
        Map the UniProt ID to the corresponding Auth Asym ID.
        """
        chain_uni_map = {}
        for i in range( df.shape[0] ):
            auth_asym_id = df.loc[i, "Auth Asym ID"]
            uni_id = df.loc[i, "Uniprot ID"]

            for aa_id in auth_asym_id.split( "," ):
                if pdb_id in self.select_chains:
                    if aa_id not in self.select_chains[pdb_id]:
                        continue
                chain_uni_map[aa_id] = {}
                chain_uni_map[aa_id]["uni_id"] = uni_id.split( "," )
        return chain_uni_map


    def get_pdb_uni_mapping( self, pdb_id: str ) -> Tuple[bool, pd.DataFrame]:
        """
        Map PDb to UniProt using SIFTS.
        """
        print( f"SIFTS mapping for {pdb_id}..." )
        sifts_file_path = os.path.join( self.misc_sifts_dir, f"{pdb_id}.tsv" )
        if os.path.exists( sifts_file_path  ):
            success = True
        else:
            success = download_SIFTS_Uni_PDB_mapping( self.misc_sifts_dir,
                                                        pdb_id,
                                                        max_trials = 10, wait_time = 5 )
            subprocess.call( ["mv", f"./{pdb_id}.tsv", sifts_file_path] )
        
        if success:
            mapping = pd.read_csv( sifts_file_path, sep = "\t", header = None )
        else:
            mapping = pd.DataFrame( {} )
        return success, mapping


    def remove_nulls( self, pdb_pos: List, uni_pos: List ):
        """
        Given a list of PDB and UniProt residue positions, remove "nulls" where present.
        """
        null_pos = [i for i, e in enumerate( pdb_pos ) if math.isnan( e ) or e == "null"]

        pdb_pos = [e for i, e in enumerate( pdb_pos ) if i not in null_pos]
        uni_pos = [e for i, e in enumerate( uni_pos ) if i not in null_pos]
        return pdb_pos, uni_pos


    def crop_residues( self, selection: np.array, pdb_pos: List, uni_pos: List ):
        """
        Select only the required residues.
        """
        new_pdb_pos, new_uni_pos = [], []
        for pdb_res, uni_res in zip( pdb_pos, uni_pos ):
            if uni_res in selection:
                new_pdb_pos.append( pdb_res )
                new_uni_pos.append( uni_res )
        return new_pdb_pos, new_uni_pos


    def get_uniprot_feats( self, pdb_id: str, mapping: pd.DataFrame, chain_uni_map: Dict ) -> Dict:
        """
        Get UniProt residue positions from the SIFTS mapping for each chain.
        Select only the required residues if specified in self.select_res dict.
        """
        max_len_exceed = []
        for chain in chain_uni_map:
            uni_id = chain_uni_map[chain]["uni_id"][0]
            mapping_dict, _ = get_sifts_mapping( mapping = mapping, chain1 = chain, uni_id1 = [uni_id] )

            if uni_id != mapping_dict["uni_id"]:
                raise ValueError( "Mismatch in PDB and SIFTS UniProt ID..." )

            pdb_pos, uni_pos = self.remove_nulls( mapping_dict["pdb_pos"],
                                                    mapping_dict["uni_pos"] )
            if pdb_id in self.select_res:
                if len( self.select_res[pdb_id][chain] ) != 0:
                    pdb_pos, uni_pos = self.crop_residues( self.select_res[pdb_id][chain],
                                                            pdb_pos, uni_pos )

            if len( uni_pos ) > self.max_len:
                max_len_exceed.append( True )

            uni_start_pos = uni_pos[0]
            uni_end_pos = uni_pos[-1]
            pdb_start_pos = pdb_pos[0]
            pdb_end_pos = pdb_pos[-1]
            chain_uni_map[chain]["pdb_pos"] = pdb_pos
            chain_uni_map[chain]["pdb_res"] = [pdb_start_pos, pdb_end_pos]
            chain_uni_map[chain]["uni_res"] = [uni_start_pos, uni_end_pos]
            chain_uni_map[chain]["length"] = uni_end_pos - uni_start_pos + 1
        return chain_uni_map, any( max_len_exceed )


    def download_pdb_struct( self, pdb_id: str ) -> bool:
        """
        Given the PDB ID download the structure file in .cif format.
        """
        pdb_file_path = os.path.join( self.misc_pdb_struct_dir, f"{pdb_id}.cif" )

        if not os.path.exists( pdb_file_path ):
            ext, _ = download_pdb( pdb_id, max_trials = 5, wait_time = 5, return_id = True )
            if ext == None:
                success = False
            else:
                
                subprocess.call( ["mv", f"./{pdb_id}.{ext}", pdb_file_path] )
                success = True
        else:
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
                            pdb_path = self.misc_pdb_struct_dir )
        
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

        print( f"Entry ID: {uni_id_pair}" )
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


    def get_misc_entries( self ):
        """
        Prepare input file for the specified PDB IDs.
        Get UniProt IDs.
        Map PDB to UniProt using SIFTS.
        Get the UniProt start, end residue positions from SIFTS mapping.
        Remove if:
            PDB has non-protein chains too.
            No SIFTS mapping.
            Any chain in PDB exceed max length.
            Mismatch in seq length and cmap dim.
        """
        misc_data = {"entry_id": [], "cmap": {}, "seq": {}, "pdb": {}, "acc": {}}

        # PDB IDs from PEDS + SV's gold mine
        from_peds = ["2jwn", "2dt7", "2n3a", "2jss", "2mkr"]
        from_SVs_gold_mine = ["1dt7", "2ruk", "2lmq", "8cmk", "2mwy", "2kqs", "5xv8"]
        from_prev = ["7lna", "6xmn"]
        for i, pdb_id in enumerate( from_peds + from_SVs_gold_mine + from_prev ):
        # for i, pdb_id in enumerate( ["5xv8"] ):
            print( f"\n-----------------> {pdb_id}" )

            df = self.get_data_from_pdb_api( pdb_id )
            print( pdb_id, "  ", df.shape )
            # Should have 2 or more protein chains.
            if df.shape[0] < 2:
                print( f"Too few chains in {pdb_id}..." )
                continue

            chain_uni_map = self.get_chain_uniprot_mapping( pdb_id, df )

            if self.too_many_chains( chain_uni_map ):
                print( "Too many chains..." )
                continue

            success, mapping = self.get_pdb_uni_mapping( pdb_id )
            if not success:
                print( "Mapping does not exist..." )
                continue
            chain_uni_map, max_len_exceed = self.get_uniprot_feats( pdb_id, mapping, chain_uni_map )

            if max_len_exceed:
                print( f"{pdb_id} exceds max length..." )
                continue

            struct_success = self.download_pdb_struct( pdb_id )
            if not struct_success:
                print( "Could not download the structure..." )
                continue

            seq_success = self.download_uniprot_seq( chain_uni_map )
            if not seq_success:
                print( "Unable to download UniProt seq..." )
                continue

            contact_map = self.create_contact_maps( pdb_id, chain_uni_map )

            success_size = self.check_seq_cmap_size( chain_uni_map, contact_map )
            if not success_size:
                print( "Mismatch in seq length and cmap size..." )
                continue

            # if np.count_nonzero( contact_map ) == 0:
            #     print( f"No contacts present at 8Angstorm in PDB: {pdb_id}..." )
            #     continue
            uni_id_pair = self.create_uni_id_pairs( chain_uni_map )
            prot1_seq, prot2_seq = self.get_prot_seq( uni_id_pair )
            misc_data["entry_id"].append( uni_id_pair )
            misc_data["cmap"][uni_id_pair] = contact_map
            # print( np.count_nonzero( contact_map )/contact_map.size )
            misc_data["seq"][uni_id_pair] = [prot1_seq, prot2_seq]
            misc_data["pdb"][uni_id_pair] = pdb_id
            misc_data["acc"][uni_id_pair] = {
                            "pdb": pdb_id,
                            "chain_uni_map": chain_uni_map
            }

        print( "\n" )
        print( "Total PDB IDs obtained: ", len( misc_data["entry_id"] ) )

        return misc_data


    def write_summary_file( self, acc_dict: Dict ):
        """
        Write relevant info for all selected entries to a .txt summary file.
        """
        summary_dict = {}
        for idx, uni_id_pair in enumerate( acc_dict ):
            pdb_id = acc_dict[uni_id_pair]["pdb"]
            chain_uni_map = acc_dict[uni_id_pair]["chain_uni_map"]

            summary_dict[uni_id_pair] = {}
            for chain in chain_uni_map:
                uni_res = chain_uni_map[chain]["uni_res"]
                pdb_res = chain_uni_map[chain]["pdb_res"]

                summary_dict[uni_id_pair]["pdb_id"] = pdb_id
                summary_dict[uni_id_pair][chain] = {
                                        "uni_res": uni_res,
                                        "pdb_res": pdb_res
                }
        with open( self.complexes_summary_file, "w" ) as w:
            json.dump( summary_dict, w, indent = 4 )


        # w = open( self.complexes_summary_file, "w" )
        # w.writelines( "-------------- case study complexes --------------\n\n" )
        # for idx, uni_id_pair in enumerate( acc_dict ):
        #     pdb_id = acc_dict[uni_id_pair]["pdb"]
        #     chain_uni_map = acc_dict[uni_id_pair]["chain_uni_map"]

        #     w.writelines( f"{idx}. {uni_id_pair} --> {pdb_id}\n" )

        #     for chain in chain_uni_map:
        #         uni_res = chain_uni_map[chain]["uni_res"]
        #         uni_res = "-".join( map( str, uni_res ) )
        #         pdb_res = chain_uni_map[chain]["pdb_res"]
        #         pdb_res = "-".join( map( str, pdb_res ) )

        #         w.writelines( f"\n\t{chain} \t PDB res = {pdb_res} \t UniProt res = {uni_res}" )
        #     w.writelines( "\n\n" )
        # w.close()


    def create_af2_input( self, misc_data: Dict ):
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
            for entry_id in misc_data["seq"].keys():
                prot1_seq, prot2_seq = misc_data["seq"][entry_id]

                uni_id1, uni_id2 = entry_id.split( "--" )
                uni_id2, copy_num = uni_id2.split( "_" )
                uni_id1, s1, e1 = uni_id1.split( ":" )
                uni_id2, s2, e2 = uni_id2.split( ":" )

                entry_id = f"{uni_id1}--{uni_id2}_{copy_num}"

                with open( f"{self.af2_fasta_dir}{entry_id}.fasta", "w" ) as w:
                    w.writelines( ">" + f"{uni_id1}:{s1}:{e1}" + "\n" + prot1_seq + "\n\n" )
                    w.writelines( ">" + f"{uni_id2}:{s2}:{e2}" + "\n" + prot2_seq + "\n" )

                w_path.writelines( f"./AF2_fasta_misc/{entry_id}.fasta," )
        w_path.close()


    def create_af3_input( self, misc_data: Dict ):
        """
        Create JSON file for batch running AF3 server.
        Will create batches of 20 merged binary complexes.
        AF3 requires a list of dict in JSON format 
                (https://github.com/google-deepmind/alphafold/blob/main/server/README.md).
        The entry_id serves as the job name.
        This allows to upload 20 jobs on the AF3 server at once,
                but you still need to run each job one by one.
        """
        misc_entry_ids = list( misc_data["seq"].keys() )
        for start in np.arange( 0, len( misc_entry_ids ), 30 ):
            if ( start + 30 ) > len( misc_entry_ids ):
                end = len( misc_entry_ids )
            else:
                end = start + 30
            af3_batch = []
            for entry_id in misc_entry_ids[start:end]:
                prot1_seq, prot2_seq = misc_data["seq"][entry_id]

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


    def save( self, misc_data: Dict ):
        self.write_summary_file( misc_data["acc"] )

        with open( self.misc_pdbs_file, "w" ) as w:
            w.writelines( ",".join( sorted( misc_data["pdb"].values() ) ) )

        with open( self.misc_uni_seq_file, "w" ) as w:
            json.dump( self.uni_seq_dict, w )

        with open( self.misc_test_input, "w" ) as w:
            w.writelines( ",".join( misc_data["entry_id"] ) )

        hf = h5py.File( self.misc_test_target, "w" )
        for k in misc_data["cmap"]:
            print( k, " -->\t", misc_data["pdb"][k], "\t",
                    misc_data["cmap"][k].shape, "\t",
                    np.count_nonzero( misc_data["cmap"][k] ), " \t",
                    round( np.count_nonzero( misc_data["cmap"][k] )/misc_data["cmap"][k].size, 3 ) )
            hf.create_dataset( k, data = misc_data["cmap"][k] )


################################################
if __name__ == "__main__":
    EntryFromPDB().forward()

"""
【中文解析-可演化改动建议】
1) 将硬编码路径迁移到统一配置文件（YAML/ENV）并加入路径合法性校验。
2) 为每个中间文件增加 schema 校验（字段名/类型/范围），降低跨脚本耦合导致的隐式错误。
3) 增加小样本 smoke test（例如 3~5 条记录）和数据快照测试，保障重构时行为一致。
4) 若面向预训练，建议输出 token 级别元信息（来源数据库、结构置信度、对齐覆盖率）便于构建多任务标签。
"""
