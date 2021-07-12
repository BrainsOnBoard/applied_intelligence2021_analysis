#include "common.h"

// BoB robotics includes
#define EXPOSE_INFOMAX_INTERNALS
#include "navigation/infomax.h"

// Standard C++ includes
#include <algorithm>

int
bobMain(int, char **)
{
    using namespace BoBRobotics::Navigation;

    const std::vector<int> numHiddens = { 5, 10, 25, 50, 100 };
    ExperimentData expt{ "num_hiddens", numHiddens };
    for (auto numHidden : numHiddens) {
        const auto weights = InfoMax<>::getInitialWeights(ImageSize.area(), numHidden, 42);
        InfoMaxRotater<> infomax{ ImageSize, weights };
        trainAndTest(infomax, expt);
    }

    return EXIT_SUCCESS;
}
