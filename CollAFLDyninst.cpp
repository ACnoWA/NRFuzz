#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstddef>
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>
#include "config.h"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#include <map>
#include <sstream>
#include <climits>
#include <set>
using namespace std;

#include "instConfig.h"
#include "instUnmap.h"
// DyninstAPI includes
#include "BPatch.h"
#include "BPatch_binaryEdit.h"
#include "BPatch_flowGraph.h"
#include "BPatch_function.h"
#include "BPatch_point.h"


using namespace Dyninst;

//hash table length

// static u32 num_conditional, // the number of total conditional edges
//             num_indirect,   // the number of total indirect edges
//             max_map_size, // the number of all edges, including potential indirect edges
static u32 AllEdge_id = 0; // assign unique id for each conditional edges

static u32 num_predtm = 0, // the number of total pre-determined edges; unique id
    num_indirect = 0,   // the number of total indirect edges
    max_map_size = 0; // the number of all edges, including potential indirect edges

//unordered_map是表示哈希表的数据结构，第一个参数是key，第二个参数是value，第三个参数是哈希函数。unorderMap是采用拉链法解决哈希冲突的。
std::unordered_map<EDGE, u32, HashEdge> cond_map;
std::unordered_map<EDGE, u32, HashEdge> condnot_map;
std::unordered_map<EDGE, u32, HashEdge> uncond_map;
std::unordered_map<EDGE, u32, HashEdge> nojump_map;
std::unordered_map<u32, vector<u32>> edgeRla_map;

//cmd line options
char *originalBinary;
char *instrumentedBinary;
bool verbose = false;
bool isPrep = false; //preprocessing
string out_path;
const string outfile = "NearedgeInfo.txt";
// set<string> instrumentLibraries;
// set<string> runtimeLibraries;


bool flag = false; //determine whether 0 id is fake or real

// call back functions
BPatch_function *ConditionJump;
BPatch_function *IndirectEdges;
BPatch_function *initAflForkServer;


const char *instLibrary = "./libCollAFLDyninst.so";

static const char *OPT_STR = "i:o:v";
static const char *USAGE = " -i <binary> -o <binary>\n \
    Analyse options:\n \
            -i: Input binary \n \
            -o: Output binary\n \
            -v: Verbose output\n \
			-P: The initial preprocessing (counting edhes and blocks; writing address files.)\n";

bool parseOptions(int argc, char **argv)
{

    int c;
    while ((c = getopt (argc, argv, OPT_STR)) != -1) {
        switch ((char) c) {
        case 'i':
            originalBinary = optarg;
            break;
        case 'o':
            instrumentedBinary = optarg;
            break;
        case 'v':
            verbose = true;
            break;
       	case 'P':
			isPrep = true;
			break;
		default:
            cerr << "Usage: " << argv[0] << USAGE;
            return false;
        }
    }

    if(originalBinary == NULL) {
        cerr << "Input binary is required!"<< endl;
        cerr << "Usage: " << argv[0] << USAGE;
        return false;
    }

    if((instrumentedBinary == NULL) && (isPrep == false)) {
        cerr << "Output binary or -P is required!" << endl;
        cerr << "Usage: " << argv[0] << USAGE;
        return false;
    }

    return true;
}

BPatch_function *findFuncByName (BPatch_image * appImage, char *funcName)
{
    BPatch_Vector < BPatch_function * >funcs;

    if (NULL == appImage->findFunction (funcName, funcs) || !funcs.size ()
        || NULL == funcs[0]) {
        cerr << "Failed to find " << funcName << " function." << endl;
        return NULL;
    }

    return funcs[0];
}

//skip some functions
bool isSkipFuncs(char* funcName){
    if (string(funcName) == string("first_init") ||
        string(funcName) == string("__mach_init") ||
        string(funcName) == string("_hurd_init") ||
        string(funcName) == string("_hurd_preinit_hook") ||
        string(funcName) == string("doinit") ||
        string(funcName) == string("doinit1") ||
        string(funcName) == string("init") ||
        string(funcName) == string("init1") ||
        string(funcName) == string("_hurd_subinit") ||
        string(funcName) == string("init_dtable") ||
        string(funcName) == string("_start1") ||
        string(funcName) == string("preinit_array_start") ||
        string(funcName) == string("_init") ||
        string(funcName) == string("fini") ||
        string(funcName) == string("_fini") ||
        string(funcName) == string("_hurd_stack_setup") ||
        string(funcName) == string("_hurd_startup") ||
        string(funcName) == string("register_tm_clones") ||
        string(funcName) == string("deregister_tm_clones") ||
        string(funcName) == string("frame_dummy") ||
        string(funcName) == string("__do_global_ctors_aux") ||
        string(funcName) == string("__do_global_dtors_aux") ||
        string(funcName) == string("__libc_csu_init") ||
        string(funcName) == string("__libc_csu_fini") ||
        string(funcName) == string("start") ||
        string(funcName) == string("_start") || 
        string(funcName) == string("__libc_start_main") ||
        string(funcName) == string("__gmon_start__") ||
        string(funcName) == string("__cxa_atexit") ||
        string(funcName) == string("__cxa_finalize") ||
        string(funcName) == string("__assert_fail") ||
        string(funcName) == string("_dl_start") || 
        string(funcName) == string("_dl_start_final") ||
        string(funcName) == string("_dl_sysdep_start") ||
        string(funcName) == string("dl_main") ||
        string(funcName) == string("_dl_allocate_tls_init") ||
        string(funcName) == string("_dl_start_user") ||
        string(funcName) == string("_dl_init_first") ||
        string(funcName) == string("_dl_init")) {
        return true; //skip these functions
        }
    return false;    
}



// instrument at conditional edges, like afl
bool instrumentCondition(BPatch_binaryEdit * appBin, BPatch_function * instFunc, BPatch_point * instrumentPoint, 
         Dyninst::Address block_addr, u32 cond_id){
    vector<BPatch_snippet *> cond_args;
    BPatch_constExpr CondID(cond_id);
    cond_args.push_back(&CondID);

    BPatch_funcCallExpr instCondExpr(*instFunc, cond_args);

    BPatchSnippetHandle *handle =
            appBin->insertSnippet(instCondExpr, *instrumentPoint, BPatch_callBefore, BPatch_firstSnippet);
    if (!handle) {
            cerr << "Failed to insert instrumention in basic block at offset 0x" << hex << block_addr << endl;
            return false;
        }
    return true;         

}


/*
num_all_edges: the number of all edges
num_condition_edges: the number of all conditional edges
ind_addr_file: path to the file that contains (src_addr des_addr id)
*/
bool instrumentIndirect(BPatch_binaryEdit * appBin, BPatch_function * instFunc, 
                BPatch_point * instrumentPoint, Dyninst::Address src_addr, u32 edge_id){
    vector<BPatch_snippet *> ind_args;

    BPatch_constExpr EdgeID(edge_id);
    ind_args.push_back(&EdgeID);
    

    BPatch_funcCallExpr instIndirect(*instFunc, ind_args);

    BPatchSnippetHandle *handle =
            appBin->insertSnippet(instIndirect, *instrumentPoint, BPatch_callBefore, BPatch_firstSnippet);
    
    if (!handle) {
            cerr << "Failed to insert instrumention in basic block at offset 0x" << hex << src_addr << endl;
            return false;
        }
    return true;

}


/*instrument at edges
    addr_id_file: path to the file that contains (src_addr des_addr id)
*/
bool edgeInstrument(BPatch_binaryEdit * appBin, BPatch_image *appImage, 
                    vector < BPatch_function * >::iterator funcIter, char* funcName){
    BPatch_function *curFunc = *funcIter;
    BPatch_flowGraph *appCFG = curFunc->getCFG ();

    BPatch_Set < BPatch_basicBlock * > allBlocks;
    if (!appCFG->getAllBasicBlocks (allBlocks)) {
        cerr << "Failed to find basic blocks for function " << funcName << endl;
        return false;
    } else if (allBlocks.size () == 0) {
        cerr << "No basic blocks for function " << funcName << endl;
        return false;
	}


    set < BPatch_basicBlock *>::iterator bb_iter;
	BPatch_basicBlock *src_bb = NULL;
	BPatch_basicBlock *trg_bb = NULL;
	unsigned long src_addr = 0;
	unsigned long trg_addr = 0;
	string cond_addr_ids = out_path + "/" + COND_ADDR_ID;
	string condnot_addr_ids = out_path + "/" + COND_NOT_ADDR_ID;
	string instru_file = out_path + "/instrument_file.txt"; 
	ofstream CondTaken_file, CondNot_file, NoJump_file, UncondJump_file, instru_stream;
//open函数，将数据流和文件关联起来，实现对文件的读写。函数的第二个参数表示打开文件的方式：out表示为输出打开文件(写文件操作)，app表示以添加的方式进行写，binary是二进制方式
	CondTaken_file.open (cond_addr_ids.c_str(), ios::out | ios::app | ios::binary); //write file
    CondNot_file.open (condnot_addr_ids.c_str(), ios::out | ios::app | ios::binary); //write file
	instru_stream.open (instru_file.c_str(), ios::out | ios::app | ios::binary); //write file	
    
	instru_stream << funcName << endl;

	for (bb_iter = allBlocks.begin (); bb_iter != allBlocks.end (); bb_iter++){
        BPatch_basicBlock * block = *bb_iter;
        vector<pair<Dyninst::InstructionAPI::Instruction, Dyninst::Address> > insns;
        block->getInstructions(insns);

        Dyninst::Address addr = insns.back().second;  //addr: equal to offset when it's binary rewrite
        Dyninst::InstructionAPI::Instruction insn = insns.back().first; 
        Dyninst::InstructionAPI::Operation op = insn.getOperation();
        Dyninst::InstructionAPI::InsnCategory category = insn.getCategory();
        Dyninst::InstructionAPI::Expression::Ptr expt = insn.getControlFlowTarget();

        //pre-determined edges
        vector<BPatch_edge *> outgoingEdge;
        (*bb_iter)->getOutgoingEdges(outgoingEdge);
        vector<BPatch_edge *>::iterator edge_iter;
        //u8 edge_type;

        for(edge_iter = outgoingEdge.begin(); edge_iter != outgoingEdge.end(); ++edge_iter) {
            //edge_type = (*edge_iter)->getType();
            //Map address and id of edges
			src_bb = (*edge_iter)->getSource();
            trg_bb = (*edge_iter)->getTarget();
			src_addr = src_bb->getStartAddress();
			trg_addr = trg_bb->getStartAddress();
			
			if ((*edge_iter)->getType() == CondJumpTaken){
				instru_stream << src_addr << " " << trg_addr << " " << AllEdge_id << endl; 	
				if (CondTaken_file.is_open()) {
					CondTaken_file << src_addr << " " << trg_addr << " " << AllEdge_id << endl; 	
				} else {
					cout << "cannot open the file: " << cond_addr_ids.c_str() << endl;
					return false;	
				}
                instrumentCondition(appBin, ConditionJump, (*edge_iter)->getPoint(), addr, AllEdge_id);
				AllEdge_id++;
                if (AllEdge_id >= MAP_SIZE) AllEdge_id = random() % MAP_SIZE;
            }
            else if ((*edge_iter)->getType() == CondJumpNottaken){
				instru_stream << src_addr << " " << trg_addr << " " << AllEdge_id << endl; 	
				if (CondNot_file.is_open()) {
					CondNot_file << src_addr << " " << trg_addr << " " << AllEdge_id << endl; 	
				} else {
					cout << "cannot open the file: " << condnot_addr_ids.c_str() << endl;
					return false;
				}
				instrumentCondition(appBin, ConditionJump, (*edge_iter)->getPoint(), addr, AllEdge_id);
                AllEdge_id++;
                if (AllEdge_id >= MAP_SIZE) AllEdge_id = random() % MAP_SIZE;
            }
            else if ((*edge_iter)->getType() == NonJump){
				instrumentCondition(appBin, ConditionJump, (*edge_iter)->getPoint(), addr, AllEdge_id);
                AllEdge_id++;
                if (AllEdge_id >= MAP_SIZE) AllEdge_id = random() % MAP_SIZE;
            } 
            else if ((*edge_iter)->getType() == UncondJump){
                instrumentCondition(appBin, ConditionJump, (*edge_iter)->getPoint(), addr, AllEdge_id);
                AllEdge_id++;
                if (AllEdge_id >= MAP_SIZE) AllEdge_id = random() % MAP_SIZE;
            }  
            
        }


        //indirect edges. Because we can't get src_addr's  location and trg_addr's location directly. So we don't record.
        for(Dyninst::InstructionAPI::Instruction::cftConstIter iter = insn.cft_begin(); iter != insn.cft_end(); ++iter) {
            if(iter->isIndirect) {
                
                if(category == Dyninst::InstructionAPI::c_CallInsn) {//indirect call
                    vector<BPatch_point *> callPoints;
                    appImage->findPoints(addr, callPoints); //use callPoints[0] as the instrument point
                    instrumentIndirect(appBin, IndirectEdges, callPoints[0], addr,  AllEdge_id);
                    AllEdge_id++;
                    if (AllEdge_id >= MAP_SIZE) AllEdge_id = random() % MAP_SIZE;
                }
                else if(category == Dyninst::InstructionAPI::c_BranchInsn) {//indirect jump
                    vector<BPatch_point *> callPoints;
                    appImage->findPoints(addr, callPoints); //use callPoints[0] as the instrument point
                    instrumentIndirect(appBin, IndirectEdges, callPoints[0], addr, AllEdge_id);
                    AllEdge_id++;
                    if (AllEdge_id >= MAP_SIZE) AllEdge_id = random() % MAP_SIZE;
                                
                }
                else if(category == Dyninst::InstructionAPI::c_ReturnInsn) {
                    vector<BPatch_point *> retPoints;
                    appImage->findPoints(addr, retPoints);

                    instrumentIndirect(appBin, IndirectEdges, retPoints[0], addr, AllEdge_id);
                    AllEdge_id++;
                    if (AllEdge_id >= MAP_SIZE) AllEdge_id = random() % MAP_SIZE;
                }
 
            }
        }
    }

    CondTaken_file.close();
    CondNot_file.close();
    //NoJump_file.close();
    //UncondJump_file.close();
	instru_stream.close(); 
	return true;
}

bool readAddrs() {
	
	string cond_addr_ids = out_path + "/" + COND_ADDR_ID;
	string condnot_addr_ids = out_path + "/" + COND_NOT_ADDR_ID;
	
	ifstream CondTaken_file, CondNot_file;
    
    /*recover addresses, ids*/
    struct stat inbuff;
    u64 src_addr, trg_addr;
    u32 edge_id;
    /*     condition taken edges   */
    if (stat(cond_addr_ids.c_str(), &inbuff) == 0){ // file  exists
        CondTaken_file.open (cond_addr_ids.c_str()); //read file
        if (CondTaken_file.is_open()){
            while (CondTaken_file >> src_addr >> trg_addr >> edge_id){
                cond_map.insert(make_pair(EDGE(src_addr, trg_addr), edge_id));
            }
            CondTaken_file.close();
        }

    }
    else{
        cout << "Please create address-ids first." <<endl;
        return false;
    }

    /*     condition not taken edges   */
    if (stat(condnot_addr_ids.c_str(), &inbuff) == 0){ // file  exists
        CondNot_file.open (condnot_addr_ids.c_str()); //read file
        if (CondNot_file.is_open()){
            while (CondNot_file >> src_addr >> trg_addr >> edge_id){
                condnot_map.insert(make_pair(EDGE(src_addr, trg_addr), edge_id));
            }
            CondNot_file.close();
        }

    }
    else{
        cout << "Please create address-ids first." <<endl;
        return false;
    }

    /*    unconditional jumps  
    if (stat(unjump_addr_ids.c_str(), &inbuff) == 0){ // file  exists
        UncondJump_file.open (unjump_addr_ids.c_str()); //read file
        if (UncondJump_file.is_open()){
            while (UncondJump_file >> src_addr >> trg_addr >> edge_id){
                uncond_map.insert(make_pair(EDGE(src_addr, trg_addr), edge_id));
            }
            UncondJump_file.close();
        }

    }
    else{
        cout << "Please create address-ids first." <<endl;
        return false;
    }

        no jumps  
    if (stat(nojump_addr_ids.c_str(), &inbuff) == 0){ // file  exists
        NoJump_file.open (nojump_addr_ids.c_str()); //read file
        if (NoJump_file.is_open()){
            while (NoJump_file >> src_addr >> trg_addr >> edge_id){
                nojump_map.insert(make_pair(EDGE(src_addr, trg_addr), edge_id));
            }
            NoJump_file.close();
        }

    }
    else{
        cout << "Please create address-ids first." <<endl;
        return false;
    }*/

    /* the number of pre-determined edges, map_size */
    /*
	fs::path num_file = output_dir / NUM_EDGE_FILE;
    ifstream NunFile;
    if (stat(num_file.c_str(), &inbuff) == 0){ // file  exists
        NunFile.open (num_file.c_str()); //read file
        if (NunFile.is_open()){
            NunFile >> max_map_size >> num_predtm;
            NunFile.close();
        }

    }
    else{
        cout << "Please create num_edges.txt first." <<endl;
        return false;
    }
	*/
	return true;
}

/* Arg: BPatch_edge 
 * Return: edge_id
 * if flag = false && edge_id = 0, the id of the given edge does not exist.
 */
u32 get_edgeId(vector <BPatch_edge *>::iterator edgeIter) {
	flag = false;
	BPatch_edge *curEdge = *edgeIter;
	BPatch_basicBlock *src_bb = NULL;
	BPatch_basicBlock *trg_bb = NULL;
	unsigned long src_addr = 0;
	unsigned long trg_addr = 0;
	std::unordered_map< EDGE, u32, HashEdge >::iterator itdl;
	u32 edge_id = 0;
	
	string rcd_file = out_path + "/rcd_file.txt";
	ofstream rcd_stream;
   	rcd_stream.open(rcd_file.c_str(), ios::out | ios::app);	

	src_bb = curEdge->getSource();
	trg_bb = curEdge->getTarget();
	src_addr = src_bb->getStartAddress();
	trg_addr = trg_bb->getStartAddress();
	
	if (curEdge->getType() == CondJumpTaken) {
		rcd_stream << src_addr << " " << trg_addr << endl;
		itdl = cond_map.find(EDGE(src_addr, trg_addr));			                 
		if (itdl != cond_map.end()){
			edge_id = (*itdl).second;
			flag = true;		
		} else {
			cout << "get_edgeId" << endl;
			cout << src_addr << " " << trg_addr << endl;
			cout << "Couldn't find a CondJumpTaken edge at address: " << src_addr << ", " << trg_addr << endl;
		}
	} else if (curEdge->getType() == CondJumpNottaken) {
		rcd_stream << src_addr << " " << trg_addr << endl;
		itdl = condnot_map.find(EDGE(src_addr, trg_addr));
		if (itdl != condnot_map.end()) {
			edge_id = (*itdl).second;
			flag = true;		
		} else {
			cout << "get_edgeId" << endl;
			cout << src_addr << " " << trg_addr << endl;
			cout << "Couldn't find a CondJumpNottaken edge at address: " << src_addr << ", " << trg_addr << endl;
		}
	}/* else if (curEdge->getType() == UncondJump) {
		rcd_stream << src_addr << " " << trg_addr << endl;
		itdl = uncond_map.find(EDGE(src_addr, trg_addr));
		if (itdl != uncond_map.end()) {
			edge_id = (*itdl).second;
			flag = true;		
		} else {
			cout << "get_edgeId" << endl;
			cout << src_addr << " " << trg_addr << endl;
			cout << "Couldn't find an UncondJump edge at address: " << src_addr << ", " << trg_addr << endl;
		}
	} else if (curEdge->getType() == NonJump) {
		rcd_stream << src_addr << " " << trg_addr << endl;
		itdl = nojump_map.find(EDGE(src_addr, trg_addr));
		if (itdl != nojump_map.end()) {
			edge_id = (*itdl).second;
			flag = true;		
		} else {
			cout << "get_edgeId" << endl;
			cout << src_addr << " " << trg_addr << endl;
			cout << "Couldn't find a NonJump edge at address: " << src_addr << ", " << trg_addr << endl;
		}
	}
	*/
	rcd_stream.close();
	return edge_id;
}


/* record all edges' near edge(parent edges and son edges) 
 * 
 */
bool rcdNearEdges(vector< BPatch_function * >& allFunctions) {
	
	/*--------------
	 * Store the address and id information in the file into map
	 * -------------
	 */
	u32 edge_id;
	ofstream edge_number;
	string out_dir = out_path + "/" + outfile;
	vector < BPatch_edge * > comingEdge;
	vector < BPatch_edge * > sonEdge;
	vector < u32 > nEdges;
	vector < BPatch_function * >::iterator funcIter;
	vector<BPatch_edge *>::iterator cmedge_iter;

	/*-----
	 * Records neighboring edge information for all edges.
	 *-----
	 */
	string rcd_file = out_path + "/rcd_file.txt";
	ofstream rcd_stream;
	rcd_stream.open(rcd_file.c_str(), ios::out | ios::app);

	edge_number.open(out_dir.c_str(), ios::out | ios::app);
	for (funcIter = allFunctions.begin(); funcIter != allFunctions.end(); ++funcIter) {
		BPatch_function *curFunc = *funcIter;
		BPatch_flowGraph *appCFG = curFunc->getCFG();
		BPatch_Set < BPatch_basicBlock * > allBlocks;
        char funcName[1024];
        curFunc->getName (funcName, 1024);
		if (!appCFG->getAllBasicBlocks (allBlocks)) {
			cerr << "Failed to find basic blocks for function " << funcName << endl;
			return false;
		} else if (allBlocks.size () == 0) {
			cerr << "No basic blocks for function " << funcName << endl;
			return false;
		}
	    
		rcd_stream << funcName << endl; 
        		
		set < BPatch_basicBlock *>::iterator bb_iter;
		for (bb_iter = allBlocks.begin (); bb_iter != allBlocks.end (); bb_iter++){
			//pre-determined edges
			vector<BPatch_edge *> outgoingEdge;
			(*bb_iter)->getOutgoingEdges(outgoingEdge);
			vector<BPatch_edge *>::iterator edge_iter;
			//u8 edge_type;
			
			comingEdge.clear();
			u32 nedge_id;
				
			(*bb_iter)->getIncomingEdges(comingEdge);
			
			for(edge_iter = outgoingEdge.begin(); edge_iter != outgoingEdge.end(); ++edge_iter) {
				edge_id = get_edgeId(edge_iter);	
				if ( flag ) {
			
					vector<BPatch_edge *>::iterator cmedge_iter;
					nEdges.clear();
					//record parent' edges	
					for(cmedge_iter = comingEdge.begin(); cmedge_iter != comingEdge.end(); ++cmedge_iter) {
						nedge_id = get_edgeId(cmedge_iter);
						if ( flag ) {
							nEdges.push_back(nedge_id);
						}
					}

					
					BPatch_basicBlock *trg_bb = (*edge_iter)->getTarget();	
					sonEdge.clear();
					trg_bb->getOutgoingEdges(sonEdge);
					//record son's edge
					for(cmedge_iter = sonEdge.begin(); cmedge_iter != sonEdge.end(); ++cmedge_iter) {
						nedge_id = get_edgeId(cmedge_iter);
						if ( flag ) {
							nEdges.push_back(nedge_id);
						}
					}
					
					//去重	
					sort(nEdges.begin(), nEdges.end());
					nEdges.erase(unique(nEdges.begin(), nEdges.end()), nEdges.end());
					
					if (!nEdges.empty()) {
						edge_number << edge_id << " ";
						copy(nEdges.begin(), nEdges.end(), ostream_iterator<u32>(edge_number, " "));
						edge_number << endl;
					}

				}
			
			}
		}	
	}

	edge_number.close();
	rcd_stream.close();

	return true;
}


/* insert forkserver at the beginning of main
    funcInit: function to be instrumented, i.e., main

*/

bool insertForkServer(BPatch_binaryEdit * appBin, BPatch_function * instIncFunc,
                         BPatch_function *funcInit)
{

    /* Find the instrumentation points */
    vector < BPatch_point * >*funcEntry = funcInit->findPoint (BPatch_entry);

    if (NULL == funcEntry) {
        cerr << "Failed to find entry for function. " <<  endl;
        return false;
    }

    //cout << "Inserting init callback." << endl;
    BPatch_Vector < BPatch_snippet * >instArgs; 

    BPatch_funcCallExpr instIncExpr(*instIncFunc, instArgs);

    /* Insert the snippet at function entry */
    BPatchSnippetHandle *handle =
        appBin->insertSnippet (instIncExpr, *funcEntry, BPatch_callBefore, BPatch_firstSnippet);
    if (!handle) {
        cerr << "Failed to insert init callback." << endl;
        return false;
    }
    return true;
}

void clearPre(){
	string cond_addr_ids = out_path + "/" + COND_ADDR_ID;
	string condnot_addr_ids = out_path + "/" + COND_NOT_ADDR_ID;
	string nojump_addr_ids = out_path + "/" + NO_JUMP_ADDR_ID;
	string unjump_addr_ids = out_path + "/" + UNCOND_JUMP_ADDR_ID;
	string out_dir = out_path + "/" + outfile; 
	//check if the file of recorded information exists
    if ((access(cond_addr_ids.c_str(), 0)) != -1) {
		if (remove(cond_addr_ids.c_str()) == 0) {
			cout << "Remove the " << cond_addr_ids << " successfully." << endl; 
		}
	}
    if ((access(condnot_addr_ids.c_str(), 0)) != -1) {
		if (remove(condnot_addr_ids.c_str()) == 0) {
			cout << "Remove the " << condnot_addr_ids << " successfully." << endl; 
		}
	}
    if ((access(nojump_addr_ids.c_str(), 0)) != -1) {
		if (remove(nojump_addr_ids.c_str()) == 0) {
			cout << "Remove the " << nojump_addr_ids << " successfully." << endl; 
		}
	}
    if ((access(unjump_addr_ids.c_str(), 0)) != -1) {
		if (remove(unjump_addr_ids.c_str()) == 0) {
			cout << "Remove the " << unjump_addr_ids << " successfully." << endl; 
		}
	}
    if ((access(out_dir.c_str(), 0)) != -1) {
		if (remove(out_dir.c_str()) == 0) {
			cout << "Remove the " << out_dir << " successfully." << endl; 
		}
	}

}

int main (int argc, char **argv){

     if(!parseOptions(argc,argv)) {
        return EXIT_FAILURE;
    }

	string str(instrumentedBinary);
	size_t found_pos = str.find_last_of("/");
	out_path = str.substr(0, found_pos);
	string out_dir = out_path + "/" + outfile;
    
	/* start instrumentation*/
    BPatch bpatch;
    // skip all libraries unless -l is set
    BPatch_binaryEdit *appBin = bpatch.openBinary (originalBinary, false);
    if (appBin == NULL) {
        cerr << "Failed to open binary" << endl;
        return EXIT_FAILURE;
    }

	//check if the file of recorded information exists
    clearPre();

	// if(!instrumentLibraries.empty()){
    //     for(auto lbit = instrumentLibraries.begin(); lbit != instrumentLibraries.end(); lbit++){
    //         if (!appBin->loadLibrary ((*lbit).c_str())) {
    //             cerr << "Failed to open instrumentation library " << *lbit << endl;
    //             cerr << "It needs to be located in the current working directory." << endl;
    //             return EXIT_FAILURE;
    //         }
    //     }
    // }

    BPatch_image *appImage = appBin->getImage ();

    
    vector < BPatch_function * > allFunctions;
    appImage->getProcedures(allFunctions);

    if (!appBin->loadLibrary (instLibrary)) {
        cerr << "Failed to open instrumentation library " << instLibrary << endl;
        cerr << "It needs to be located in the current working directory." << endl;
        return EXIT_FAILURE;
    }

    initAflForkServer = findFuncByName (appImage, (char *) "initAflForkServer");
 
    ConditionJump = findFuncByName (appImage, (char *) "ConditionJump");
    IndirectEdges = findFuncByName (appImage, (char *) "IndirectEdges");


    if (!initAflForkServer || !ConditionJump || !IndirectEdges) {
        cerr << "Instrumentation library lacks callbacks!" << endl;
        return EXIT_FAILURE;
    }
    
	/* count the number of edges for the length of hash table
    1. num_c = the number of conditional edges
    2. num_i = the number of indirect call/jump sites
    3. length of hash table = num_c + num_i
    */
	/*
	if (isPrep) {
		string num_file = out_path + "/" + NUM_EDGE_FILE;
		num_predtm = 0;
		num_indirect = 0;
		max_map_size = 0;

		for (auto countIter = allFunctions.begin(); countIter != allFunctions.end(); ++countIter) {
			BPatch_function *countFunc = *countIter;
			ParseAPI::Function *f = ParseAPI::convert(countFunc);
		 	// We should only instrument functions in .text.
            ParseAPI::CodeRegion* codereg = f->region();
            ParseAPI::SymtabCodeRegion* symRegion = dynamic_cast<ParseAPI::SymtabCodeRegion*>(codereg);
            assert(symRegion);
            SymtabAPI::Region* symR = symRegion->symRegion();
            if (symR->getRegionName() != ".text")
                continue;

            char funcName[1024];
            countFunc->getName (funcName, 1024);
            
            if(isSkipFuncs(funcName)) continue;
            //count edges
            if(!count_edges(appBin, appImage, countIter, funcName, out_dir)) 
            	cout << "Empty function" << funcName << endl;	
		}	
	    // fuzzer gets the number of edges by saved file
        
        u32 num_tpm = num_predtm + num_indirect * BASE_INDIRECT;
        u16 num_exp = (u16)ceil( log(num_tpm) / log(2) );
        // be general with the shared memory
        if(num_exp < MAP_SIZE_POW2) num_exp = MAP_SIZE_POW2;


        max_map_size = (1 << num_exp);
        
        ofstream numedges;
        numedges.open (num_file.c_str(), ios::out | ios::app | ios::binary); //write file
        if(numedges.is_open()){
            numedges << max_map_size << " " << num_predtm << endl; 
            //numedges << num_indirect << endl;
        }
        numedges.close();    
        //TODO: fuzzer gets the values through pipe (or shared memory?)?
        return EXIT_SUCCESS; 

	}*/
   /* instrument edges */
    vector < BPatch_function * >::iterator funcIter;
    for (funcIter = allFunctions.begin (); funcIter != allFunctions.end (); ++funcIter) {
        BPatch_function *curFunc = *funcIter;
        char funcName[1024];
        curFunc->getName (funcName, 1024);
        //if(isSkipFuncs(funcName)) continue;
        //instrument at edges
        //if(!edgeInstrument(appBin, appImage, funcIter, funcName, addr_id_file)) return EXIT_FAILURE;
        edgeInstrument(appBin, appImage, funcIter, funcName);
    }
	
	if (!readAddrs()) {
		cout << "Failed to read Address." << endl;
		return EXIT_FAILURE;
	}	

	if (!rcdNearEdges(allFunctions)) {
		cout << "Failed to record near side information." << endl;
	} else {
		cout << "Recoreded near side information successfully." << endl;
	}

	BPatch_function *funcToPatch = NULL;
    BPatch_Vector<BPatch_function*> funcs;
    
    appImage->findFunction("_start",funcs);
    if(!funcs.size()) {
        cerr << "Couldn't locate _start, check your binary. "<< endl;
        return EXIT_FAILURE;
    }
    // there should really be only one
    funcToPatch = funcs[0];

    if(!insertForkServer (appBin, initAflForkServer, funcToPatch)){
        cerr << "Could not insert init callback at _start." << endl;
        return EXIT_FAILURE;
    }

    if(verbose){
        cout << "Saving the instrumented binary to " << instrumentedBinary << "..." << endl;
    }
    // save the instrumented binary
    if (!appBin->writeFile (instrumentedBinary)) {
        cerr << "Failed to write output file: " << instrumentedBinary << endl;
        return EXIT_FAILURE;
    }

    if(verbose){
        cout << "All done! Happy fuzzing!" << endl;
    }

    return EXIT_SUCCESS;


}
