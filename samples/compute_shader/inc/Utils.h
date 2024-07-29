#pragma once
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>

namespace Utils{
std::vector<char> readFile(const std::string& filename);

};