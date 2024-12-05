#include <iostream>
#include <vector>
#include <cmath>
#include <random>
struct Point {
    double x, y;

    double distanceTo(const Point& other) const {
        return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
    }
};

struct Node {
    Point point;
    Node* parent;

    Node(const Point& p, Node* pr = nullptr) : point(p), parent(pr) {}
};
class RRT {
private:
    std::vector<Node*> nodes;
    Point lower_bound, upper_bound;
    Point start, goal;
    double step_size;
    double goal_sample_rate;  // 概率以目标点为采样点
    int max_iterations;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis_x, dis_y, dis_goal;

public:
    RRT(const Point& start, const Point& goal, const Point& lower_bound,
        const Point& upper_bound, double step_size, double goal_sample_rate, int max_iterations)
        : start(start), goal(goal), lower_bound(lower_bound), upper_bound(upper_bound),
          step_size(step_size), goal_sample_rate(goal_sample_rate), max_iterations(max_iterations),
          gen(rd()), dis_x(lower_bound.x, upper_bound.x), dis_y(lower_bound.y, upper_bound.y), dis_goal(0, 1) {
        nodes.push_back(new Node(start));
    }

    ~RRT() {
        for (Node* node : nodes) {
            delete node;
        }
    }

    Point randomPoint() {
        if (dis_goal(gen) < goal_sample_rate) {
            return goal;
        } else {
            return {dis_x(gen), dis_y(gen)};
        }
    }

    Node* nearestNode(const Point& point) {
        Node* nearest = nullptr;
        double min_dist = std::numeric_limits<double>::max();
        for (auto node : nodes) {
            double dist = node->point.distanceTo(point);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = node;
            }
        }
        return nearest;
    }

    Node* extend(Node* nearest, const Point& random_point) {
        Point direction = {random_point.x - nearest->point.x, random_point.y - nearest->point.y};
        double length = sqrt(direction.x * direction.x + direction.y * direction.y);
        Point new_point;
        if (length > step_size) {
            new_point.x = nearest->point.x + (direction.x / length) * step_size;
            new_point.y = nearest->point.y + (direction.y / length) * step_size;
        } else {
            new_point = random_point;
        }
        Node* new_node = new Node(new_point, nearest);
        nodes.push_back(new_node);
        return new_node;
    }

    bool run() {
        for (int i = 0; i < max_iterations; ++i) {
            Point rnd_point = randomPoint();
            Node* nearest = nearestNode(rnd_point);
            Node* new_node = extend(nearest, rnd_point);
            if (new_node->point.distanceTo(goal) <= step_size) {
                return true;  // 成功找到路径
            }
        }
        return false;  // 未能找到路径
    }
};


int main() {
    Point lower_bound = {0, 0};
    Point upper_bound = {100, 100};
    Point start = {5, 5};
    Point goal = {95, 95};
    double step_size = 5;
    double goal_sample_rate = 0.1;  // 10% 概率直接选取目标点
    int max_iterations = 10000000;

    RRT rrt(start, goal, lower_bound, upper_bound, step_size, goal_sample_rate, max_iterations);
    bool found_path = rrt.run();

    if (found_path) {
        std::cout << "Path to goal found!" << std::endl;
    } else {
        std::cout << "No path to goal found." << std::endl;
    }

    return 0;
}
