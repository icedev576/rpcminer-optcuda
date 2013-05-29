FIND_PACKAGE(CURL)

#set(CMAKE_BUILD_TYPE Debug)

IF(CURL_FOUND)
	INCLUDE_DIRECTORIES(${CURL_INCLUDE_DIRS})
ENDIF(CURL_FOUND)

SET(BITCOIN_RPC_MINER_SRC
	${CMAKE_SOURCE_DIR}/src/sha256.cpp
	${CMAKE_SOURCE_DIR}/src/rpcminer/hex.cpp
	${CMAKE_SOURCE_DIR}/src/rpcminer/httprequest.cpp
	${CMAKE_SOURCE_DIR}/src/rpcminer/rpcminerclient.cpp
	${CMAKE_SOURCE_DIR}/src/rpcminer/rpcminermain.cpp
	${CMAKE_SOURCE_DIR}/src/rpcminer/rpcrequest.cpp
	${CMAKE_SOURCE_DIR}/src/json/json_spirit_reader.cpp
	${CMAKE_SOURCE_DIR}/src/json/json_spirit_value.cpp
	${CMAKE_SOURCE_DIR}/src/json/json_spirit_writer.cpp
	${CMAKE_SOURCE_DIR}/src/minercommon/base64.c
	${CMAKE_SOURCE_DIR}/src/cryptopp/cpu.cpp
	${CMAKE_SOURCE_DIR}/src/cryptopp/sha.cpp
)

SET(BITCOIN_RPC_MINER_CUDA_SRC
	${CMAKE_SOURCE_DIR}/src/rpcminer/rpcminerthreadgpu.cpp
	${CMAKE_SOURCE_DIR}/src/cuda/bitcoinminercuda.cpp
	${CMAKE_SOURCE_DIR}/src/cuda/bitcoinminercuda.cu
)

SET(BITCOIN_RPC_MINER_OPENCL_SRC
	${CMAKE_SOURCE_DIR}/src/rpcminer/rpcminerthreadgpu.cpp
	${CMAKE_SOURCE_DIR}/src/opencl/bitcoinmineropencl.cpp
)

IF(BITCOIN_ENABLE_CUDA)
	ADD_DEFINITIONS(-D_BITCOIN_MINER_CUDA_)
	
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -O3  -Xptxas=-v )
#-Xopencc=-LIST:source=on -Xptxas=-v --keep -O4)


	CUDA_ADD_EXECUTABLE(rpcminer ${BITCOIN_RPC_MINER_SRC} ${BITCOIN_RPC_MINER_CUDA_SRC})
	# Install generated PTX CUDA module
	INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/${generated_file_basename}.ptx" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/rpcminer-cuda" RENAME "bitcoinminercuda.ptx")
	# Install rpcminer
	INSTALL(TARGETS rpcminer DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/rpcminer-cuda")
ELSE(BITCOIN_ENABLE_CUDA)
	IF(BITCOIN_ENABLE_OPENCL)
		ADD_DEFINITIONS(-D_BITCOIN_MINER_OPENCL_)
		ADD_EXECUTABLE(rpcminer ${BITCOIN_RPC_MINER_SRC} ${BITCOIN_RPC_MINER_OPENCL_SRC})
		INSTALL(FILES "${CMAKE_SOURCE_DIR}/src/opencl/bitcoinmineropencl.cl" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/rpcminer-opencl")
    # Install rpcminer
    INSTALL(TARGETS rpcminer DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/rpcminer-opencl")
	ENDIF(BITCOIN_ENABLE_OPENCL)
ENDIF(BITCOIN_ENABLE_CUDA)

TARGET_LINK_LIBRARIES(rpcminer ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} ${CURL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

IF(BITCOIN_ENABLE_CUDA)
	TARGET_LINK_LIBRARIES(rpcminer ${CUDA_LIBRARIES})
ENDIF(BITCOIN_ENABLE_CUDA)

IF(BITCOIN_ENABLE_OPENCL)
	TARGET_LINK_LIBRARIES(rpcminer ${OPENCL_LIBRARY})
ENDIF(BITCOIN_ENABLE_OPENCL)
