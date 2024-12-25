#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

cd ${ScriptPath}/Common
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
make install

cd ${ScriptPath}/DVPPLite
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
make install

cd ${ScriptPath}/Media
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
make install

cd ${ScriptPath}/OMExecute
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
make install

echo "build all so." 
