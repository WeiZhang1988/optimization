set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
include_directories(${PROJECT_SOURCE_DIR}/../include
										${PROJECT_SOURCE_DIR}/../third_party/eigen
										${PROJECT_SOURCE_DIR}/../third_party/matplot)
add_executable(app app.cpp)
target_include_directories(app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app PUBLIC pthread)
install(TARGETS app DESTINATION ${PROJECT_SOURCE_DIR}/exe)
