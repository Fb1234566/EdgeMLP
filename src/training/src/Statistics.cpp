#include "../include/Statistics.h"
#include <stdexcept>

std::vector<double> TrainingStatics::operator()(const std::string stat) const
{
    if (stats.count(stat)<1)
    {
        throw std::invalid_argument(std::string("Invalid static: ") + stat);
    }
    return stats.at(stat);
}

void TrainingStatics::operator()(const std::string stat, const double value)
{
    if (stats.count(stat)<1)
    {
        stats[stat] = std::vector<double>();
    }
    stats[stat].push_back(value);
}

std::vector<std::string> TrainingStatics::getStatsPreset() const
{
    std::vector<std::string> keys;
    keys.reserve(stats.size());
    for (const auto& pair : stats) {
        keys.push_back(pair.first);
    }
    return keys;
}

std::ostream& operator<< (std::ostream& os, const TrainingStatics& s)
{
    for (const auto& key : s.getStatsPreset()) {
        os << key << ": ";
        std::vector<double> values = s(key);
        for (size_t i = 0; i < values.size(); ++i) {
            os << values[i];
            if (i != values.size() - 1)
                os << ", ";
        }
        os << std::endl;
    }
    return os;
}
