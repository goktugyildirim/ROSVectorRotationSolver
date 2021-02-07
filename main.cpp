#include <iostream>
#include <ros/ros.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>

#include <cmath>

#include <string>
#include <iostream>
#include <fstream>
#include "openGA.hpp"


using std::string;
using std::cout;
using std::endl;


ros::Publisher pub_marker_arr_;
std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> vector_pairs;


Eigen::Matrix3d
giveRotationMatrix(double yaw, double pitch, double roll)
{
    Eigen::Matrix3d trans;
    trans.setIdentity();

    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3d R = q.matrix();

    trans.topLeftCorner(3,3) = R;
    return trans;
}

struct MySolution
{
    double yaw;
    double pitch;
    double roll;

    string to_string() const
    {
        return
                string("{")
                +  "yaw:"+std::to_string(yaw)
                +", pitch:"+std::to_string(pitch)
                +", roll:"+std::to_string(roll)
                +"}";
    }
};

struct MyMiddleCost
{
    // This is where the results of simulation
    // is stored but not yet finalized.
    double objective1;
};

typedef EA::Genetic<MySolution,MyMiddleCost> GA_Type;
typedef EA::GenerationType<MySolution,MyMiddleCost> Generation_Type;

void init_genes(MySolution& p,const std::function<double(void)> &rnd01)
{
    // rnd01() gives a random number in 0~1
    p.yaw=0.0+6.283*rnd01();
    p.pitch=0.0+6.283*rnd01();
    p.roll=0.0+6.283*rnd01();
}

bool eval_solution(
        const MySolution& p,
        MyMiddleCost &c)
{
    const double& yaw=p.yaw;
    const double& pitch=p.pitch;
    const double& roll=p.roll;

    Eigen::Matrix3d mat_rotate = giveRotationMatrix(yaw,pitch,roll);

    double cost = 0;

    for(const auto& pair:vector_pairs)
    {
        auto vector_source = pair.first;
        auto vector_target = pair.second;

        auto vector_estimated = mat_rotate*vector_source;

        auto vector_diff = vector_estimated-vector_target;

        double single_cost = pow(vector_diff[0],2) +  pow(vector_diff[1],2) +  pow(vector_diff[2],2);

        cost += pow(single_cost,2);

    }

    c.objective1=cost/vector_pairs.size();
    return true; // solution is accepted
}

MySolution mutate(
        const MySolution& X_base,
        const std::function<double(void)> &rnd01,
        double shrink_scale)
{
    MySolution X_new;
    const double mu = 0.2*shrink_scale; // mutation radius (adjustable)
    bool in_range;
    do{
        in_range=true;
        X_new=X_base;
        X_new.yaw+=mu*(rnd01()-rnd01());
        in_range=in_range&&(X_new.yaw>=0.0 && X_new.yaw<6.283);
        X_new.pitch+=mu*(rnd01()-rnd01());
        in_range=in_range&&(X_new.pitch>=0.0 && X_new.pitch<6.283);
        X_new.roll+=mu*(rnd01()-rnd01());
        in_range=in_range&&(X_new.roll>=0.0 && X_new.roll<6.283);
    } while(!in_range);
    return X_new;
}

MySolution crossover(
        const MySolution& X1,
        const MySolution& X2,
        const std::function<double(void)> &rnd01)
{
    MySolution X_new;
    double r;
    r=rnd01();
    X_new.yaw=r*X1.yaw+(1.0-r)*X2.yaw;
    r=rnd01();
    X_new.pitch=r*X1.pitch+(1.0-r)*X2.pitch;
    r=rnd01();
    X_new.roll=r*X1.roll+(1.0-r)*X2.roll;
    return X_new;
}

double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    // finalize the cost
    double final_cost=0.0;
    final_cost+=X.middle_costs.objective1;
    return final_cost;
}

std::ofstream output_file;

void SO_report_generation(
        int generation_number,
        const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,
        const MySolution& best_genes)
{
    cout
            <<"Generation ["<<generation_number<<"], "
            <<"Best="<<last_generation.best_total_cost<<", "
            <<"Average="<<last_generation.average_cost<<", "
            <<"Best genes=("<<best_genes.to_string()<<")"<<", "
            <<"Exe_time="<<last_generation.exe_time
            <<endl;


}

void solve()
{

    EA::Chronometer timer;
    timer.tic();

    GA_Type ga_obj;
    ga_obj.problem_mode=EA::GA_MODE::SOGA;
    ga_obj.multi_threading=true;
    ga_obj.idle_delay_us=10; // switch between threads quickly
    ga_obj.dynamic_threading=true;
    ga_obj.verbose=false;
    ga_obj.population=1000;
    ga_obj.generation_max=1000;
    ga_obj.calculate_SO_total_fitness=calculate_SO_total_fitness;
    ga_obj.init_genes=init_genes;
    ga_obj.eval_solution=eval_solution;
    ga_obj.mutate=mutate;
    ga_obj.crossover=crossover;
    ga_obj.SO_report_generation=SO_report_generation;
    ga_obj.crossover_fraction=0.7;
    ga_obj.mutation_rate=0.2;
    ga_obj.best_stall_max=10;
    ga_obj.elite_count=10;
    ga_obj.solve();

    cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<endl;
}






Eigen::Matrix4d
giveTransformationMatrix(const double &dx, const double &dy, const double &dz, const double &yaw,
                                                const double &pitch, const double &roll)
{
    Eigen::Matrix4d trans;
    trans.setIdentity();

    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3d R = q.matrix();

    trans.topLeftCorner(3,3) = R;

    trans(0,3) = dx;
    trans(1,3) = dy;
    trans(2,3) = dz;

    return trans;
}

visualization_msgs::Marker
GiveArrowmarker(Eigen::Vector3d vec, const std::string& frame_id, const int& id, std::string type)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time();
    marker.ns = "arrow";
    marker.id = id;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.02;
    marker.scale.y = 0.04;
    marker.scale.z = 0.06;

    if(type == "source")
    {
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
    }else{
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
    }



    geometry_msgs::Point start, end;
    start.x = 0;
    start.y = 0;
    start.z = 0;
    end.x = start.x + vec.x();
    end.y = start.y + vec.y();
    end.z = start.z + vec.z();
    marker.points.resize(2);
    marker.points[0].x = start.x;
    marker.points[0].y = start.y;
    marker.points[0].z = start.z;
    marker.points[1].x = end.x;
    marker.points[1].y = end.y;
    marker.points[1].z = end.z;
    return marker;
}


Eigen::Vector3d
DefineVector(double x, double y, double z)
{Eigen::Vector3d vector; vector << x, y, z; return  vector;}


std::pair<Eigen::Vector3d, Eigen::Vector3d>
generateUnitVectorPair(Eigen::Vector3d vector_source, double yaw, double pitch, double roll, visualization_msgs::MarkerArray& markerArray,
                       int pair_id)
{
    visualization_msgs::Marker arrow_source = GiveArrowmarker(vector_source.normalized(),"world", pair_id+1, "source");
    markerArray.markers.push_back(arrow_source);

    Eigen::Matrix3d mat_rotate = giveRotationMatrix(yaw,pitch,roll);

    Eigen::Vector3d vector_target = mat_rotate*vector_source;
    visualization_msgs::Marker arrow_target = GiveArrowmarker(vector_target.normalized(),"world", pair_id+2, "target");
    markerArray.markers.push_back(arrow_target);

    std::pair<Eigen::Vector3d, Eigen::Vector3d> pair(vector_source.normalized(), vector_target.normalized());
    return pair;
}

void ShowEstimations(double yaw, double pitch, double roll, visualization_msgs::MarkerArray& markerArray)
{
    Eigen::Matrix3d mat_rotate = giveRotationMatrix(yaw,pitch,roll);
    int id = 100;

    for(auto pair:vector_pairs)
    {
        id += 1;
        auto source = pair.first;
        auto est = mat_rotate*source;

        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time();
        marker.ns = "arrow";
        marker.id = id;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.02;
        marker.scale.y = 0.04;
        marker.scale.z = 0.06;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;

        geometry_msgs::Point start, end;
        start.x = 0;
        start.y = 0;
        start.z = 0;
        end.x = start.x + est.normalized().x();
        end.y = start.y + est.normalized().y();
        end.z = start.z + est.normalized().z();
        marker.points.resize(2);
        marker.points[0].x = start.x;
        marker.points[0].y = start.y;
        marker.points[0].z = start.z;
        marker.points[1].x = end.x;
        marker.points[1].y = end.y;
        marker.points[1].z = end.z;
        markerArray.markers.push_back(marker);
    }


}


void Callback(const ros::TimerEvent &event)
{
    visualization_msgs::MarkerArray markerArray;

    Eigen::Vector3d vec1_s = DefineVector(5,1e-15,1e-15);
    auto pair1 = generateUnitVectorPair(vec1_s, 1, 1, 0, markerArray, 1);

    Eigen::Vector3d vec2_s = DefineVector(1e-15,5,1e-15);
    auto pair2 = generateUnitVectorPair(vec2_s, 1, 1, 0, markerArray, 6);

    Eigen::Vector3d vec3_s = DefineVector(1e-15,1e-15,-5);
    auto pair3 = generateUnitVectorPair(vec3_s, 1, 1, 0, markerArray, 10);

    Eigen::Vector3d vec4_s = DefineVector(1e-15,-6,5);
    auto pair4 = generateUnitVectorPair(vec4_s, 1, 1, 0, markerArray, 18);

    Eigen::Vector3d vec5_s = DefineVector(7,6,-5);
    auto pair5 = generateUnitVectorPair(vec5_s, 1, 1, 0, markerArray, 30);


    vector_pairs.push_back(pair1);
    vector_pairs.push_back(pair2);
    vector_pairs.push_back(pair3);
    vector_pairs.push_back(pair4);
    vector_pairs.push_back(pair5);

    //solve();
    ShowEstimations(2.141662, 4.141101, 3.141260, markerArray);




    pub_marker_arr_.publish(markerArray);

    std::cout << "************************************************" << std::endl;
}



int main(int argc, char** argv) {

    ros::init (argc, argv, "rotation_solver_node");
    ros::NodeHandle nh;


    ros::Timer timer = nh.createTimer(ros::Duration(0.5), Callback);
    pub_marker_arr_ = nh.advertise<visualization_msgs::MarkerArray>("/vectors",0);


    ros::spin ();
    ros::waitForShutdown();
    return 0;
}
