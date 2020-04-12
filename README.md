# Overview

##
Tested on a machine with Nvidia 2080Ti, Ubuntu 16.04/18.04, Tensorflow 1.8.0 and Keras 2.2.3.<br/>
- Python 2.7
- Tensorflow
- Keras

# Install dyninst
We use Dyninst to instrument target binaries. So firstly, install Dyninst [the branch](https://github.com/mxz297/dyninst).<br/>
For the branch of dyninst, use `csifuzz`.
## step1
```bash
mkdir dyninst101
cd dyninst101
root_dir=`pwd`
```
## step2 Install capstone
```bash
git clone https://github.com/mxz297/capstone.git thirdparty/capstone
cd thirdparty/capstone
git checkout access-fixes
cd $root_dir
cd thirdparty/capstone
mkdir install
mkdir -p build
cd build

# Configure
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../install ..

# Install
nprocs=`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`
make -j "$(($nprocs / 2))" install
```
## step3 Install libunwind
```bash
cd $root_dir
git clone  https://github.com/mxz297/libunwind.git thirdparty/libunwind
cd thirdparty/libunwind
mkdir install
# Configure
./autogen.sh
./configure --prefix=`pwd`/install --enable-cxx-exceptions

# Install
nprocs=`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`
make -j "$(($nprocs / 2))" install
```
## step4 Install Dyninst
```bash
cd $root_dir
git clone https://github.com/mxz297/dyninst.git thirdparty/dyninst-10.1.0
cd thirdparty/dyninst-10.1.0/
git checkout csifuzz
cd $root_dir
cd thirdparty/dyninst-10.1.0/
mkdir install
mkdir -p build
cd build

# Configure
cmake -DLibunwind_ROOT_DIR=`pwd`/../../libunwind/install -DCapstone_ROOT_DIR=`pwd`/../../capstone/install/ -DCMAKE_INSTALL_PREFIX=`pwd`/../install -G 'Unix Makefiles' ..

nprocs=`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`
make -j "$(($nprocs / 2))"
# Build
#   Dyninst build tends to succeed with a retry after an initial build failure.
#   Cover that base with couple of retries.

make install
```
# Set up Envs
```bash
export DYNINST_INSTALL=/path/to/dyninst/install/dir
export NRFUZZ_PATH=/path/to/nrfuzz

export DYNINSTAPI_RT_LIB=$DYNINST_INSTALL/lib/libdyninstAPI_RT.so
export LD_LIBRARY_PATH=$DYNINST_INSTALL/lib:$NRFUZZ_PATH
export PATH=$PATH:$NRFUZZ_PATH
```
# Build
```bash
cd /path/to/nrfuzz
make
./CollAFLDyninst -i /path/to/need/instruct/binary -o /path/to/instructed/binary
```
## Usage
After instrumenation, copy `NearedgeInfo.txt`,`instructed_binary`from /path/to/instructed/binary to ./programs/×××/. Then copy `new_nn.py`,`neuzz`,`afl-showmap`,`libCollAFLDyninst.so` to ./programs/×××/.<br>

After completing this, We use a sample program readelf as an example to demonstrate how to execute.<br/>

Open a terminal, start nn module
```bash
    #python nn.py [program [arguments]]
    python nn.py ./readelf -a
```
open another terminal, start neuzz module.
```bash
    #./neuzz -i in_dir -o out_dir -l mutation_len [program path [arguments]] @@
    ./neuzz -i neuzz_in -o seeds -l 7506 ./readelf -a @@  
```
## Sample programs
Try 10 real-world programs on NRFuzz. Check setup details at programs/[program names]/README.




