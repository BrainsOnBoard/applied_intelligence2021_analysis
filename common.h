#pragma once

// BoB robotics includes
#include "common/path.h"
#include "common/stopwatch.h"
#include "navigation/image_database.h"

// OpenCV
#include <opencv2/opencv.hpp>

// Standard C++ includes
#include <iostream>
#include <vector>

static const cv::Size ImageSize{ 90, 25 };
static const auto DatabaseRoot = BoBRobotics::Path::getProgramDirectory() /
                                 "../datasets/rc_car/Stanmer_park_dataset";
static const std::vector<std::string> TrainRoutes = { "0511/unwrapped_dataset1",
                                                      "0511/unwrapped_dataset2" };
const std::vector<std::string> TestRoutes = { "0511/unwrapped_dataset3" };

inline void
loadDatabaseImages(std::vector<cv::Mat> &images,
                   const std::string &dbName)
{
    const BoBRobotics::Navigation::ImageDatabase db{ DatabaseRoot / dbName };
    BOB_ASSERT(!db.empty());

    db.loadImages(images, ImageSize);
    std::cout << "Loaded " << images.size() << " images from " << dbName << "\n";
}

inline auto
loadDatabaseImages(const std::vector<std::string> &dbNames)
{
    std::vector<cv::Mat> images;
    for (const auto &dbName : dbNames) {
        loadDatabaseImages(images, dbName);
    }

    return images;
}

template<class Algo>
void
doTest(Algo &algo, const std::vector<cv::Mat> &testImages, cv::FileStorage &fs)
{
    static std::vector<double> headings;
    headings.clear();
    BoBRobotics::Stopwatch timer;

    std::cout << "Testing...";
    timer.start();
    for (const auto &image : testImages) {
        const units::angle::degree_t heading = std::get<0>(algo.getHeading(image));
        headings.push_back(heading.value());
    }
    const units::time::millisecond_t testTime = timer.elapsed();
    std::cout << "Completed in " << testTime << "\n";

    fs << "time_per_image_ms" << testTime.value() / testImages.size()
       << "headings_deg"
       << headings;
}

struct ExperimentData {
    template<class... Ts>
    ExperimentData(Ts&&... params)
      : fs{ BoBRobotics::Path::getProgramPath().str() + ".json",
            cv::FileStorage::WRITE }
      , trainImages{ loadDatabaseImages(TrainRoutes) }
      , testImages{ loadDatabaseImages(TestRoutes) }
    {
        fs << "data"
           << "{"
           << "bob_robotics_git_commit" << BOB_ROBOTICS_GIT_COMMIT
           << "bob_project_git_commit" << BOB_PROJECT_GIT_COMMIT;
        saveExtraParams(std::forward<Ts>(params)...);
        fs << "experiments"
           << "[";
    }
    
    template<class T>
    void saveExtraParams(const std::string &name, const T &value)
    {
        fs << name << value;
    }
    
    void saveExtraParams()
    {}

    ~ExperimentData()
    {
        fs << "]" << "}";
    }

    cv::FileStorage fs;
    const std::vector<cv::Mat> trainImages, testImages;
};

template<class Algo>
void trainAndTest(Algo &algo, ExperimentData &expt)
{
    BoBRobotics::Stopwatch timer;
    std::cout << "Training...";
    timer.start();
    algo.trainRoute(expt.trainImages);
    const units::time::millisecond_t trainTime = timer.elapsed();
    std::cout << "Completed in " << trainTime << "\n";

    expt.fs << "{"
            << "image_size" << ImageSize
            << "training"
            << "{"
            << "routes"
            << TrainRoutes
            << "time_per_image_ms" << trainTime.value() / expt.trainImages.size()
            << "}";

    expt.fs << "testing"
            << "{"
            << "routes"
            << TestRoutes;
    doTest(algo, expt.testImages, expt.fs);
    expt.fs << "}";

    expt.fs << "}";
}

template<class Algo>
void trainAndTest(Algo &algo)
{
    ExperimentData expt;
    trainAndTest(algo, expt);
}
