//
// Created by hongzhe on 2/15/22.
//

/**
*  @file TestOfEntropyRegularizedMotionPlanning.cpp
*  @author Hongzhe Yu
**/


#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include <gpmp2/obstacle/ObstaclePlanarSDFFactorGPPointRobot.h>
#include <gpmp2/gp/GaussianProcessPriorLinear.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <random>

#include <iostream>

using namespace std;
using namespace gtsam;
using namespace gpmp2;
using namespace Eigen;

inline gtsam::Vector errorWrapper(const ObstaclePlanarSDFFactorGPPointRobot& factor,
                           const gtsam::Vector& conf1, const gtsam::Vector& vel1,
                           const gtsam::Vector& conf2, const gtsam::Vector& vel2) {
    return factor.evaluateError(conf1, vel1, conf2, vel2);
}

// convert sdf vector to hinge loss err vector
inline gtsam::Vector convertSDFtoErr(const gtsam::Vector& sdf, double eps) {
    gtsam::Vector err_ori = 0.0 - sdf.array() + eps;
    return (err_ori.array() > 0.0).select(err_ori, gtsam::Vector::Zero(err_ori.rows()));  // (R < s ? P : Q)
}

class GaussianSamplerSparsePrecision {
public:
    GaussianSamplerSparsePrecision(SparseMatrix<double> const &precision)
            : GaussianSamplerSparsePrecision(VectorXd::Zero(precision.rows()), precision) {}

    GaussianSamplerSparsePrecision(VectorXd const &mean, SparseMatrix<double> const &precision)
            : mean(mean), precision_(precision) {
        updatePrecisionMatrix(precision);
    }

protected:
    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;
    Eigen::SparseMatrix<double> precision_;

public:
    bool updatePrecisionMatrix(const Eigen::SparseMatrix<double>& newPrecision)
    {
        precision_ = newPrecision;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> sparsecholesky{precision_};
        transform = sparsecholesky.matrixL();
        transform = transform.inverse().transpose();
//            cout << "precision Chelosky decomp transform " << endl << transform << endl;
        return true;
    }

    bool updateMean(const Eigen::VectorXd& new_mean)
    {
        mean = new_mean;
        return true;
    }

    Eigen::MatrixXd operator()(const int &nn) const {
        cout << "precision in the sampler " << endl << precision_ << endl;
        static std::mt19937 gen{std::random_device{}()};
//        std::mt19937 gen(1);
        std::normal_distribution<> dist(0, 1);
//        return MatrixXd{transform * MatrixXd{MatrixXd::NullaryExpr(mean.size(), nn, [&](auto x) { return dist(gen); })}}; //test the fixed seed
        return (transform * Eigen::MatrixXd::NullaryExpr(mean.size(), nn, [&](auto x) { return dist(gen); })).colwise() +
               mean;
    }

};


/* ************************************************************************** */
// signed distance field data
gtsam::Matrix field, map_ground_truth;
PlanarSDF sdf;

TEST(ObstaclePlanarSDFFactorArm, data) {

map_ground_truth = (gtsam::Matrix(7, 7) <<
                                 0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,
        0,     0,     1,     1,     1,     0,     0,
        0,     0,     1,     1,     1,     0,     0,
        0,     0,     1,     1,     1,     0,     0,
        0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0).finished();
field = (gtsam::Matrix(7, 7) <<
                      2.8284,    2.2361,    2.0000,    2.0000,    2.0000,    2.2361,    2.8284,
        2.2361,    1.4142,    1.0000,    1.0000,    1.0000,    1.4142,    2.2361,
        2.0000,    1.0000,   -1.0000,   -1.0000,   -1.0000,    1.0000,    2.0000,
        2.0000,    1.0000,   -1.0000,   -2.0000,   -1.0000,    1.0000,    2.0000,
        2.0000,    1.0000,   -1.0000,   -1.0000,   -1.0000,    1.0000,    2.0000,
        2.2361,    1.4142,    1.0000,    1.0000,    1.0000,    1.4142,    2.2361,
        2.8284,    2.2361,    2.0000,    2.0000,    2.0000,    2.2361,    2.8284).finished();
// bottom-left is (0,0), length is +/- 1 per point.
Point2 origin(0, 0);
double cell_size = 1.0;

sdf = PlanarSDF(origin, cell_size, field);
// test the sdf coordinates
//gtsam::Vector2 pos{3.0,3.0};
//cout << "signed distance at " << endl << pos << " is " << endl << sdf.getSignedDistance(pos) << endl;
}

/* ************************************************************************** */
TEST(ObstaclePlanarSDFFactorGPpR, error) {
    // 2D point robot
    const int ndof = 2, nlinks = 1, nspheres = 1, nsupptd_states = 2, N = 1;
    const int ndim = 2 * ndof * nlinks * nsupptd_states;
    PointRobot pR(ndof, nlinks);

    double r = 1.5;
    BodySphereVector body_spheres;
    body_spheres.push_back(BodySphere(0, r, Point3(0.0, 0.0, 0.0)));

    SharedNoiseModel Qc_model = noiseModel::Isotropic::Sigma(2, 1.0);

    double delta_t = 0.1, tau = 0.025;
    double obs_eps = 0.2, obs_sigma = 1.0;

    PointRobotModel pRModel(pR, body_spheres);
    ObstaclePlanarSDFFactorGPPointRobot factor(0, 1, 2, 3, pRModel, sdf, obs_sigma, obs_eps,
                                               Qc_model, delta_t, tau);

    auto NoiseModel = factor.noiseModel();
//    Matrix modelInfomationMatrix = NoiseModel->R() * NoiseModel->R();
//    cout << modelInfomationMatrix << endl;

    // just check cost of two link joint
    gtsam::Vector2 q1, q2, qdot1, qdot2;
    gtsam::Matrix H1_act, H2_act, H3_act, H4_act;

    // origin zero  and stationary case
    q1 = gtsam::Vector2(0, 0);
    q2 = gtsam::Vector2(6, 4);
    qdot1 = gtsam::Vector2(0, 0);
    qdot2 = gtsam::Vector2(0, 0);

    gtsam::Vector err_act;
    err_act = factor.evaluateError(q1, qdot1, q2, qdot2, H1_act, H2_act, H3_act, H4_act);

    // optimization vars
    gtsam::Vector mu = concatVectors({q1, qdot1, q2, qdot2});
    Eigen::SparseMatrix<double> invSigma(ndim, ndim);
    int nnz = 3*ndim-2;
    invSigma.reserve(nnz);
    typedef Triplet<double> T;
    // filling sparse matrix
    vector< T > tripletList;
    tripletList.reserve(nnz);

    for (int i=0; i<ndim; i++)
    {
        tripletList.emplace_back(i,i,1);
    }
    invSigma.setFromTriplets(tripletList.begin(), tripletList.end());

    // Derivatives
    gtsam::Vector d_mu;
    gtsam::Matrix d_invSigma;

    // Gaussian sampler
    int nn = 10000; // number of samples
    GaussianSamplerSparsePrecision sampler_sparse_precision{mu, invSigma};
    MatrixXd samples(ndim, nn);

    //generate samples
    samples = sampler_sparse_precision(nn);

    gtsam::Vector losses(nn);
    gtsam::Vector Vdmu = gtsam::Vector::Zero(ndim);
    gtsam::Matrix Vddmu = gtsam::Matrix::Zero(ndim, ndim);
    double accum_loss = 0;

    for (int i = 0; i < 2; i++) {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr_solver(invSigma);
        // Calulating the expectations
        auto colwise = samples.colwise();
        std::for_each(colwise.begin(), colwise.end(), [&](auto const &elem) {
            q1 = gtsam::Vector{elem(seq(0, 1))};
            q2 = gtsam::Vector{elem(seq(2, 3))};
            qdot1 = gtsam::Vector{elem(seq(4, 5))};
            qdot2 = gtsam::Vector{elem(seq(6, 7))};
            auto hinge_loss = factor.evaluateError(q1, qdot1, q2, qdot2, H1_act, H2_act, H3_act, H4_act);
            if (hinge_loss.sum() > 0){
                cout << "positive loss " << endl;
            }
//            gtsam::Vector hinge_loss{};
            gtsam::Matrix Obs_invSigma;
            Obs_invSigma = gtsam::Matrix::Identity(hinge_loss.size(), hinge_loss.size()) / obs_sigma;
//            cout << "Obs_invSigma " << endl << Obs_invSigma << endl;
            double loss = (hinge_loss.transpose() * Obs_invSigma * hinge_loss);
            accum_loss += loss;
            Vdmu = Vdmu + (elem - mu) * loss;
            Vddmu = Vddmu + (elem - mu) * (elem - mu).transpose() * loss;
        });

        cout << "accum_loss " << accum_loss << endl;

        // Update mu and precision matrix
        d_mu = qr_solver.solve(-Vdmu);
        d_invSigma = -invSigma + Vddmu;
    }


//    // aux vectors
//    gtsam::Vector vec_b;
//
//    gtsam::Matrix H, Qc, sigmaObs, invQ1, Phi, invA, invK;

//    H = gtsam::Matrix::Zero(nspheres, 2*ndof * nlinks * nsupptd_states);
//    Qc = getQc(Qc_model);
//    sigmaObs = gtsam::Matrix::Identity(nspheres, nspheres) / obs_sigma;
//
//    invQ1 = calcQ_inv(Qc, delta_t);
//    Phi = calcPhi(ndof, delta_t);

//    invA = (gtsam::Matrix((nsupptd_states+1)*2*ndof, nsupptd_states*2*ndof)<<
//    Matrix::Identity(2*ndof, 2*ndof), gtsam::Matrix::Zero(2*ndof, 2*ndof),
//    -Phi, Matrix::Identity(2*ndof, 2*ndof),
//            gtsam::Matrix::Zero(2*ndof, 2*ndof), gtsam::Matrix::Identity(2*ndof, 2*ndof)).finished();
//
//    gtsam::Matrix invK_0 = Matrix::Identity(2*ndof, 2*ndof);
//    gtsam::Matrix invK_N = Matrix::Identity(2*ndof, 2*ndof);
//
//    gtsam::Matrix invQ = diag({invK_0, invQ1, invK_N});
//
//    cout << "invQ dim " << invQ.rows() << " " << invQ.cols() << endl;
//    cout << "invA dim " << invA.rows() << " " << invA.cols() << endl;
//
//    invK = invA.transpose() * invQ * invA;

    // parameters
    double step_size_mu = 0.001;
    double step_size_Pq = 0.001;



//    for (int i=0; i<10; i++)
//    {
//        H.setZero();
//        d_mu.setZero();
//        d_invSigma.setZero();
//
//        err_act = factor.evaluateError(q1, qdot1, q2, qdot2, H1_act, H2_act, H3_act, H4_act);
//
//        H.block(0, 0, 1, 2) = H1_act;
//        H.block(0, 2, 1, 2) = H2_act;
//        H.block(0, 4, 1, 2) = H3_act;
//        H.block(0, 6, 1, 2) = H4_act;
//
//        vec_b = H * mu - err_act;
//
//        d_mu = 2 * H.transpose() * (H * mu - vec_b);
//
//        d_invSigma = H.transpose() * sigmaObs * H + invK.transpose() - invSigma.inverse().transpose() / 2.0;
//
//        Vector l_mu = (Vector(ndim) << mu).finished();
//        Matrix l_Pq = (Matrix(ndim, ndim) << invSigma).finished();
//
//        mu = mu - step_size_mu * d_mu;
//
//        invSigma = invSigma - step_size_Pq * d_invSigma;
//
//        q1 = (Vector(2)<<sub(mu, 0, 2)).finished();
//        qdot1 = (Vector(2)<<sub(mu, 2, 4)).finished();
//        q2 = (Vector(2)<<sub(mu, 4, 6)).finished();
//        qdot2 = (Vector(2)<<sub(mu, 6, 8)).finished();
//
////        cout << "mu " << endl << mu << endl;
////        cout << "Pq " << P_q << endl;
//        cout << "derivatives: " << endl << "mu" << endl << d_mu.norm() << endl <<
//        "d_Pq" << endl << d_invSigma.norm() << endl;
//        // see the residual
//        cout << "difference wrpt last step: mu" << endl << (mu - l_mu).norm() << endl <<
//        "Pq " << endl << (invSigma - l_Pq).norm() <<endl;
//
//    }

}

/* ************************************************************************** */
/* main function */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
