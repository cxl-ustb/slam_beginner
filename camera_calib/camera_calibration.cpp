#include<Eigen/Dense>
#include<iostream>
#include<cmath>
using namespace std;
using namespace Eigen;

class SingleCamera{
public:
    SingleCamera(MatrixXf world_coor,MatrixXf pixel_coor,int n){
        this->__world_coor=world_coor;
        this->__pixel_coor=pixel_coor;
        this->__point_num=n;

        this->__P.resize(__point_num,12);
        this->__roM.resize(3,4);
        this->__A.resize(3,3);
        this->__b.resize(3,1);
        this->__K.resize(3,3);
        this->__R.resize(3,3);
        this->__t.resize(3,1);
    }

    void composeP(){
        int i=0;
        ArrayXXf P(this->__point_num,12);

        while(i<this->__point_num){
            int c=i/2;
            ArrayXf p1(4,1);
            p1=(ArrayXf)this->__world_coor.row(c);
            ArrayXf p2(4,1);
            p2<<0,0,0,0;
            if(i%2==0){
                ArrayXf p3(4,1);
                p3=-p1*this->__pixel_coor(c,0);
                P.row(i)<<p1.transpose(),p2.transpose(),p3.transpose();
            }
            else if(i%2==1){
                ArrayXf p3(4,1);
                p3=-p1*this->__pixel_coor(c,1);
                P.row(i)<<p2.transpose(),p1.transpose(),p3.transpose();
            }
            i++;
        }
        this->__P=P.matrix();
    }
    // svd to P，return A,b, where M=[A b]
    void svdP(){
    JacobiSVD<MatrixXf> svd(this->__P,ComputeThinU|ComputeThinV);
    MatrixXf U=svd.matrixU(),sigma=svd.singularValues(),V=svd.matrixV();
    MatrixXf preM=V.col(V.cols()-1);
    Map<MatrixXf> roMT(preM.data(),4,3);
    MatrixXf roM=roMT.transpose();
    MatrixXf A=roM.block(0,0,3,3);
    MatrixXf b=roM.block(0,3,3,1);
    cout<<A<<endl;
    cout<<b<<endl;
    this->__roM=roM;
    this->__A=A;
    this->__b=b;
    }

    void workInAndOut(){
        Vector3f a3T=this->__A.row(2);
        float under=a3T.norm();
        float ro01=1.0/under;

        Vector3f a1T=this->__A.row(0);
        Vector3f a2T=this->__A.row(1);

        float cx=ro01*ro01*a1T.transpose().dot(a3T),
            cy=ro01*ro01*a2T.transpose().dot(a3T);
        
        Vector3f a_cross13=a1T.cross(a3T),a_cross23=a2T.cross(a3T);

        float theta=-acos((-1.0)*(a_cross13.dot(a_cross23))/(a_cross13.norm()*a_cross23.norm())),
              alpha=ro01*ro01*a_cross13.norm()*sin(theta),
              beta=ro01*ro01*a_cross23.norm()*sin(theta);
        
        MatrixXf K(3,3);
        K<<alpha,-alpha*(1.0/tan(theta)),cx,
           0,beta/sin(theta),cy,
           0,0,1;
        this->__K=K;

        Vector3f r1=(a_cross23)/(a_cross23.norm()),
                 r301=ro01*a3T,
                r2=r301.cross(r1);
        MatrixXf R(3,3);
        R<<r1.transpose(),r2.transpose(),r301.transpose();
        this->__R=R;

        MatrixXf K_inverse=K.inverse();
        MatrixXf T=ro01*K_inverse*this->__b;
        this->__t=T;
    }

    void selfCheck(MatrixXf w_check,MatrixXf c_check){
        int my_size=w_check.rows();
        VectorXf my_err(my_size);
        for(int i=0;i<my_size;i++){
            MatrixXf tmp_p=w_check.row(i);
            tmp_p.resize(4,1);
            Vector3f  test_pix=this->__roM*tmp_p;
            int u=test_pix(0)/test_pix(2),
                v=test_pix(1)/test_pix(2),
                u_c=c_check(i,0),
                v_c=c_check(i,1);
                cout<<"第"<<i<<"个测试点的误差是"<<abs(u-u_c)+abs(v-v_c)<<endl;

        }
    }
private:
    MatrixXf __P,__roM,__A,__b,__K,__R,__t;

    MatrixXf __world_coor,__pixel_coor;
    int __point_num;
};

int main()
{
    // world corrdinate
    // points: (8, 0, 9), (8, 0, 1), (6, 0, 1), (6, 0, 9)
    MatrixXf w_xz(4,4);
    w_xz<<8, 0, 9, 1, 8, 0, 1, 1, 6, 0, 1, 1, 6, 0, 9, 1;
    // Map<MatrixXf> w_XZ(w_xz.data(),4,4);
    // points: (5, 1, 0), (5, 9, 0), (4, 9, 0), (4, 1, 0)
    MatrixXf w_xy(4,4);
    w_xy<<5, 1, 0, 1, 5, 9, 0, 1, 4, 9, 0, 1, 4, 1, 0, 1;
    //points: (0, 4, 7), (0, 4, 3), (0, 8, 3), (0, 8, 7)
    MatrixXf w_yz(4,4);
    w_yz<<0, 4, 7, 1, 0, 4, 3, 1, 0, 8, 3, 1, 0, 8, 7, 1;
    
    MatrixXf w_coor(w_xz.rows()+w_xy.rows()+w_yz.rows(),4);
    w_coor<<w_xz,w_xy,w_yz;

    // pixel coordinate
    MatrixXf c_xz(4,2);
    c_xz<<275, 142, 312, 454, 382, 436, 357, 134;
    MatrixXf c_xy(4,2);
    c_xy<<432, 473, 612, 623, 647, 606, 464, 465;
    MatrixXf c_yz(4,2);
    c_yz<<654, 216, 644, 368, 761, 420, 781, 246;
    MatrixXf c_coor(c_xz.rows()+c_xy.rows()+c_yz.rows(),2);
    c_coor<<c_xz,c_xy,c_yz;

    // coordinate for validation whether the M is correct or not
    MatrixXf w_check(5,4);
    w_check<<6, 0, 5, 1, 3, 3, 0, 1, 0, 4, 0, 1, 0, 4, 4, 1, 0, 0, 7, 1;
    MatrixXf c_check(5,2);
    c_check<<369, 297, 531, 484, 640, 468, 646, 333, 556, 194;

    SingleCamera aCamera(w_coor,c_coor,12);
    aCamera.composeP();
    aCamera.svdP();
    aCamera.workInAndOut();
    aCamera.selfCheck(w_check,c_check);

    return 0;
}
