#ifndef EDGEMLP_STATISTICS_H
#define EDGEMLP_STATISTICS_H

#include <map>
#include <string>
#include <vector>
#include <iostream>

class TrainingStatics
{
private:
    std::map<std::string, std::vector<double>> stats; // Temporarily use double as the value of the statistics
public:
    std::vector<double> operator()(std::string stat) const;
    void operator()(std::string stat, double value);
    std::vector<std::string> getStatsPreset() const;
};

std::ostream& operator << (std::ostream& os, const TrainingStatics& s);

#endif //EDGEMLP_STATISTICS_H