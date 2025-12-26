cmake_minimum_required(VERSION 3.15)
project(nexusquant VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# 源文件
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cc")
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")

# 创建库
add_library(nq SHARED ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(nq)

# OpenTelemetry C++ (optional, requires opentelemetry-cpp)
# Uncomment when OpenTelemetry C++ SDK is installed
# find_package(opentelemetry-cpp REQUIRED)
# target_link_libraries(nq PUBLIC opentelemetry::api opentelemetry::sdk)

# 安装
install(TARGETS nq
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# 测试 (optional)
enable_testing()
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

