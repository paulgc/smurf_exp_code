#include <map>
#include <vector>

class InvertedIndex {
  public:
    std::map<int, std::vector<int> > index;                       
    std::vector<int> size_vector;                                       

    InvertedIndex();
    InvertedIndex(std::map<int, std::vector<int> >&, std::vector<int>&);
    ~InvertedIndex();
    void set_fields(std::map<int, std::vector<int> >&, std::vector<int>&);  
};
