if (WIN32)
set(NVML_TOP_DIR "C:/Program Files/NVIDIA Corporation/GDK/gdk_win7_amd64_release/nvml")
else (WIN32)
set(NVML_TOP_DIR "/usr/local/cuda-9.1")
endif (WIN32)


FIND_PATH(NVML_INCLUDE_DIR nvml.h
	${NVML_TOP_DIR}/include
)

FIND_LIBRARY(NVML_DEBUG_LIBRARIES NAMES nvml
   PATHS
	 ${NVML_TOP_DIR}/lib
	 ${NVML_TOP_DIR}/lib64
)

FIND_LIBRARY(NVML_RELEASE_LIBRARIES NAMES nvml
   PATHS
	 ${NVML_TOP_DIR}/lib
	 ${NVML_TOP_DIR}/lib64
)

if(NVML_INCLUDE_DIR)
   set(NVML_FOUND TRUE)
endif(NVML_INCLUDE_DIR)
	 
if(NVML_FOUND)
   if(NOT NVML_FIND_QUIETLY)
      message(STATUS "Found NVML: ${NVML_INCLUDE_DIR}")
   endif(NOT NVML_FIND_QUIETLY)
else(NVML_FOUND)
   if(NVML_FIND_REQUIRED)
      message(FATAL_ERROR "could NOT find NVML")
   endif(NVML_FIND_REQUIRED)
endif(NVML_FOUND)

MARK_AS_ADVANCED(NVML_INCLUDE_DIR)


