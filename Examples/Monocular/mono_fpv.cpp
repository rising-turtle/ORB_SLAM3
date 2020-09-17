/*
  Sep. 17, 2020, He Zhang, hzhang8@vcu.edu

  monocular vo using FPV dataset

  TODO: add support multi sequences input

*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string& path, const string &strImgLeft, const string &strImgRight,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

double ttrack_tot = 0;


int main(int argc, char **argv)
{
    const int num_seq = 1; // (argc-3)/4;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (argc > 4); // (((argc-3) % 4) == 1);
    string file_name;
    if (bFileName)
        file_name = string(argv[argc-1]);

    if(argc < 4)
    {
        cerr << endl << "Usage: ./stereo_inertial_fpv_vi path_to_vocabulary path_to_settings path_to_data_folder (trajectory_file_name)" << endl;
        return 1;
    }

    string data_path;

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageLeftFilenames;
    vector< vector<string> > vstrImageRightFilenames;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeftFilenames.resize(num_seq);
    vstrImageRightFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        data_path = argv[3];
        string left_image_file = data_path + "/left_images.txt";
        string right_image_file = data_path + "/right_images.txt";
        // LoadImages(string(argv[4*(seq+1)-1]), string(argv[4*(seq+1)]), string(argv[4*(seq+1)+1]), vstrImageLeftFilenames[seq], vstrImageRightFilenames[seq], vTimestampsCam[seq]);
        LoadImages(data_path, left_image_file, right_image_file, vstrImageLeftFilenames[seq], vstrImageRightFilenames[seq], vTimestampsCam[seq]);
        cout << "Total images: " << vstrImageLeftFilenames[seq].size() << endl;
        cout << "Total cam ts: " << vTimestampsCam[seq].size() << endl;
        cout << std::fixed << "first cam ts: " << vTimestampsCam[seq][0] << endl;

        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageLeftFilenames[seq].size();
        tot_images += nImages[seq];

        if((nImages[seq]<=0))
        {
            cerr << "ERROR: Failed to load images sequence" << seq << endl;
            return 1;
        }
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, true);

    int proccIm = 0;
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    for (seq = 0; seq<num_seq; seq++)
    {

        // Main loop
        // cv::Mat imLeft, imRight;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {

            // Read image from file
            imLeft = cv::imread(vstrImageLeftFilenames[seq][ni],cv::IMREAD_GRAYSCALE);
            // imRight = cv::imread(vstrImageRightFilenames[seq][ni],cv::IMREAD_GRAYSCALE);

            // clahe
            clahe->apply(imLeft,imLeft);
            // clahe->apply(imRight,imRight);

            // cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
            // cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

            double tframe = vTimestampsCam[seq][ni];

            if(imLeft.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageLeftFilenames[seq][ni] << endl;
                return 1;
            }


            /*cout << "first imu: " << first_imu[seq] << endl;
            cout << "first imu time: " << fixed << vTimestampsImu[seq][0] << endl;
            cout << "size vImu: " << vImuMeas.size() << endl;*/

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the image to the SLAM system
            SLAM.TrackMonocular(imLeft,tframe);

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ttrack_tot += ttrack;
            // std::cout << "ttrack: " << ttrack << std::endl;

            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
        }
        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }


    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics

    // Save camera trajectory
    std::chrono::system_clock::time_point scNow = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(scNow);
    std::stringstream ss;
    ss << now;

    if (bFileName)
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else

    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages[0]; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages[0]/2] << endl;
    cout << "mean tracking time: " << totaltime/proccIm << endl;

    return 0;
}
// void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
//                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)

void LoadImages(const string& path, const string &strImgLeft, const string &strImgRight,
    vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    cout <<"data sequence path: "<< path << endl;
    cout <<"left image file: "<< strImgLeft << endl;
    cout <<"right image file:"<< strImgRight << endl;

    ifstream fleft, fright;
    fleft.open(strImgLeft.c_str());
    fright.open(strImgRight.c_str());

    if(!fleft.is_open() || !fright.is_open()){
      cerr << "stereo_inertial_fpv_vi.cpp: failed to open left_image_file or right_image_file, return!"<<endl;
      return;
    }

    // delete the first line
    string s;
    getline(fleft, s);
    getline(fright, s);

    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);

    while(!fleft.eof() && !fright.eof()){
      string ls, rs;
      getline(fleft, ls);
      getline(fright, rs);

      int id;
      double timestamp;
      string l_img, r_img;

      if(!ls.empty() && !rs.empty()){
        stringstream lss;
        lss << ls;
        lss >> id >> timestamp >> l_img;
        stringstream rss;
        rss << rs;
        rss >> id >> timestamp >> r_img;
        vTimeStamps.push_back(timestamp);
        vstrImageLeft.push_back(path + "/" + l_img);
        vstrImageRight.push_back(path + "/" + r_img);
      }

    }
    cout <<" stereo_inertial_fpv_vi.cpp: LoadImages have "<<vTimeStamps.size()<<" stereo images"<<endl;
    return ;
}
