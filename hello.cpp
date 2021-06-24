#include<iostream>
using namespace std;

class Complex{
private:
	double real, imag;
public:
	Complex(double r, double i):real{r},imag{i}{
	
	} 
	Complex(){
		real = 0;
		imag = 0;
	}
	
	Complex& operator+(Complex& c){
		Complex temp(real, imag);
		temp.real += c.real;
		temp.imag += c.imag;
		return temp;
	} 
	
	~Complex(){
		cout<<"deconstructor real="<<real<<" imag="<<imag<<endl;
	}
};

int main()
{
	Complex c1(1,2),c2(3,4),c3(5,6);
	Complex c4;
	c4 = c1 + c2 + c3;
	return 0;
}
