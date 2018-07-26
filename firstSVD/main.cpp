#include<iostream>
#include<fstream>
#include<string>
#include <vector>

int main(int argc, char const *argv[]) {

  std::vector<int>numbers;
  int number;
  
  while(InFile >> number)
      numbers.push_back(number);

  return 0;
}
