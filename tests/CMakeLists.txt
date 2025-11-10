cmake_minimum_required(VERSION 3.16)
project(MetalCppResidencyTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ResidencyTests
  ResidencyTestsMain.cpp
  ResidencyBudgetTests.cpp
  ProbabilisticResidencyTests.cpp
)

target_include_directories(ResidencyTests PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/.."
)
