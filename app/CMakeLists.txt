cmake_minimum_required(VERSION 3.16)
project(app VERSION 1.0.0)
set (CMAKE_CXX_STANDARD 17)

add_subdirectory(src)
install(DIRECTORY data DESTINATION ${PROJECT_SOURCE_DIR}/exe)
