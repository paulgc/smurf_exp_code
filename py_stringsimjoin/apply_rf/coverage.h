#include <vector>
#include <bitset>

#define MAX_SIZE 110000                                                               

class Coverage {
  public:
    std::bitset<MAX_SIZE> bit_vector;
    int count;
    int size;
 
    Coverage();
    Coverage(std::vector<bool>&);
    ~Coverage();

    int and_sum(const Coverage&);
    void and_coverage(const Coverage&);
    void or_coverage(const Coverage&);
    void reset();
    int sum();
};
