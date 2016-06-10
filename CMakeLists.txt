cmake_minimum_required(VERSION 2.8)
project( DRD )
find_package( OpenCV REQUIRED )
add_executable( gray GrayStampRemoval.cpp)
target_link_libraries( gray ${OpenCV_LIBS} )

