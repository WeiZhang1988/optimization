set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/lib)
set(OPTIMIZER_INCLUDE ${PROJECT_SOURCE_DIR}/include)
add_library(optimizer SHARED optimizer.cpp)
target_include_directories(optimizer BEFORE PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
																						 			  ${PROJECT_SOURCE_DIR}/include
																						 				${PROJECT_SOURCE_DIR}/third_party/eigen
																						 				${PROJECT_SOURCE_DIR}/third_party/matplot)
target_link_libraries(optimizer PUBLIC pthread)
install(TARGETS optimizer LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)
