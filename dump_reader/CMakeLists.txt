cmake_minimum_required(VERSION 3.15)
project(psoff_shaderdump)

add_executable(psoff_shaderdump
  main.cpp
)

add_dependencies(psoff_shaderdump psOff_recompiler)
target_link_libraries(psoff_shaderdump PRIVATE psOff_recompiler)