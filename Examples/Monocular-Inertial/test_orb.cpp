/*
  Sep. 30, 2020, He Zhang, hzhang8@vcu.edu

  monocular inertial using test data 

*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "ImuTypes.h"

using namespace std;

void LoadImages(const string& path, const string &strImgLeft, vector<string> &vstrImageLeft, vector<double> &vTimeStamps);
void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

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
        cerr << endl << "Usage: ./mono_inertial_test path_to_vocabulary path_to_settings path_to_data_folder (trajectory_file_name)" << endl;
        return 1;
    }

    string data_path;

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageLeftFilenames;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeftFilenames.resize(num_seq);
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
        string left_image_file = data_path + "/timestamp.txt";
        // LoadImages(string(argv[4*(seq+1)-1]), string(argv[4*(seq+1)]), string(argv[4*(seq+1)+1]), vstrImageLeftFilenames[seq], vstrImageRightFilenames[seq], vTimestampsCam[seq]);
        LoadImages(data_path, left_image_file, vstrImageLeftFilenames[seq], vTimestampsCam[seq]);
        cout << "Total images: " << vstrImageLeftFilenames[seq].size() << endl;
        cout << "Total cam ts: " << vTimestampsCam[seq].size() << endl;
        cout << std::fixed << "first cam ts: " << vTimestampsCam[seq][0] << endl;

        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        string imu_file = data_path + "/imu_vn100.log";
        // LoadIMU(string(argv[4*(seq+1)+2]), vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        LoadIMU(imu_file, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);

        cout << "Total IMU meas: " << vTimestampsImu[seq].size() << endl;
        cout << "first IMU ts: " << vTimestampsImu[seq][0] << endl;
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageLeftFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered

    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    /*cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "IMU data in the sequence: " << nImu << endl << endl;*/

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_MONOCULAR, true, 0);

    int proccIm = 0;
    cv::Mat im;
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
            im = cv::imread(vstrImageLeftFilenames[seq][ni],cv::IMREAD_GRAYSCALE);
        
            // clahe
            // clahe->apply(imLeft,imLeft);
            // clahe->apply(imRight,imRight);

            // cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
            // cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

            double tframe = vTimestampsCam[seq][ni];

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageLeftFilenames[seq][ni] << endl;
                return 1;
            }


            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0)
            {
                // cout << "t_cam " << tframe << endl;

                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni])
                {
                    // vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[first_imu],vGyro[first_imu],vTimestampsImu[first_imu]));
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    // cout << "t_imu = " << fixed << vImuMeas.back().t << endl;
                    first_imu[seq]++;
                }
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
            SLAM.TrackMonocular(im,tframe,vImuMeas);

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

    if (bFileName){
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else{
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}

void LoadImages(const string& path, const string &strImgLeft, vector<string> &vstrImageLeft, vector<double> &vTimeStamps)
{
    cout <<"data sequence path: "<< path << endl;
    cout <<"left image file: "<< strImgLeft << endl;

    ifstream fleft; 
    fleft.open(strImgLeft.c_str());
   
    if(!fleft.is_open()){
      cerr << "monocular_test.cpp: failed to open left_image_file return!"<<endl;
      return;
    }

    // delete the first line
    string s;
    getline(fleft, s);

    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    
    while(!fleft.eof()){
      string ls; 
      getline(fleft, ls);

      double timestamp;
      string l_img;

      if(!ls.empty()){
        stringstream lss;
        lss << ls;
        lss  >> timestamp >> l_img;
        vTimeStamps.push_back(timestamp);
        vstrImageLeft.push_back(path + "/" + l_img);
      }

    }
    cout <<" monocular_test.cpp: LoadImages have "<<vTimeStamps.size()<<" images"<<endl;
    return ;
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    if(!fImu.is_open()){
      cerr << "monocular_test.cpp: failed to read imu file at "<<strImuPath<<endl;
      return;
    }

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);
        if (s[0] == '#')
            continue;

        int id;
        double timestamp;
        double ax, ay, az, gx, gy, gz;
        if(!s.empty()){
          stringstream ss;
          ss << s;
          // ss >> id >> timestamp >> gx >> gy >> gz >> ax >> ay >> az;
          ss >> timestamp >> ax >> ay >> az >> gx >> gy >> gz;

          vTimeStamps.push_back(timestamp);
          vAcc.push_back(cv::Point3f(ax, ay, az));
          vGyro.push_back(cv::Point3f(gx, gy, gz));
        }
    }

    cout << "monocular_test.cpp: succeed to read "<<vTimeStamps.size()<<" imu measurements"<<endl;
    return ;
}
