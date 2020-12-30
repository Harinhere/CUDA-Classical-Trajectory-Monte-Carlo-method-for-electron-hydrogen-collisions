//Author: Harindranath Ambalampitiya, PhD(Theoretical atomic and molecular physics)
//GPU accelerated Classical Trajectory Monte Carlo Methods
//Positronium collision with proton/antiproton

#include <iostream>
#include<fstream>
#include<math.h>
#include<stdio.h>
#include<ctime>
#include<cstdlib>
#include <chrono> 
#include<curand_kernel.h>


using namespace std;
using namespace std::chrono;

// initialize random_number generator on the device
//each thread gets the same seed,but different sequence
__global__ void rng_init(curandState *state,int seed,int n)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n)
	{
		curand_init(seed, id, 0, &state[id]);
	}
} 

//root-search(bisection method) on the device

__device__ float rootxi(float alpha,float ecc,float x1,float x2)
{
	int jmax=50;
	float xacc=1e-4f;//accuracy
	float dx,xmid,rtb;
	float f=x1-ecc*sin(x1)-alpha;
	float fmid=x2-ecc*sin(x2)-alpha;
	
	rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
	for (int j=0;j<jmax;j++) 
	{
		xmid=rtb+(dx *= 0.5);
		fmid=xmid-ecc*sin(xmid)-alpha;
		if (fmid <= 0.0)
			rtb=xmid;
		if (abs(dx) < xacc || fmid == 0.0)
			return rtb;
	}
	return 0;
}

// derivatives for Runge-Kutta stepper

__device__ void derivs(float t, float *y, float *dydt)
{
	
	  float R1=sqrt(y[0]*y[0]+y[2]*y[2]+y[4]*y[4]+.2);
	  float R2=sqrt(y[6]*y[6]+y[8]*y[8]+y[10]*y[10]+.2);
	    
	  float dx=y[0]-y[6];
	  float dy=y[2]-y[8];
	  float dz=y[4]-y[10];
	  
	  float R3=sqrt(dx*dx+dy*dy+dz*dz+.2);
	  dydt[0]=y[1];
	  dydt[1]=-y[0]/powf(R1,3)+dx/powf(R3,3);
	  dydt[2]=y[3];
	  dydt[3]=-y[2]/powf(R1,3)+dy/powf(R3,3);
	  dydt[4]=y[5];
	  dydt[5]=-y[4]/powf(R1,3)+dz/powf(R3,3);
	  
	  dydt[6]=y[7];
	  dydt[7]=-y[6]/powf(R2,3)-dx/powf(R3,3);
	  dydt[8]=y[9];
	  dydt[9]=-y[8]/powf(R2,3)-dy/powf(R3,3);
	  dydt[10]=y[11];
	  dydt[11]=-y[10]/powf(R2,3)-dz/powf(R3,3);
	
}



//Let's calculate tics on the device
__global__ void ticsKernel(curandState *state,int *a,float Ev,float bstep)
{		
    int nps=3,lps=0;
	
	
	// eliptical orbit parameters
	//semi-major axis,period/2pi,Ps energy
	float asem=powf((float)nps,2.),tred=sqrtf(powf(asem,3.));
	float ehyd=-0.5/asem;
	
	float pi=2.*asinf(1.0f);
	//total energy,maximum-time iteration
	float energy=ehyd+Ev/27.21,tmax=500.;
	//initial velocity and z coordinates
	float vinit=sqrtf(Ev/27.21),zinit=-25.;
	//asymptotic z
	float zasym=25.;
	
	
	//impact parameter that each thread gets
	

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	float bimpact=bstep*(float)idx;
	
	a[idx]=0;
	//printf("blockid,threadid:%i\t%i\t%f \n",blockIdx.x,
	//threadIdx.x,bimpact);
	//copy state to local memory
	curandState localState = state[idx];
	
	// run ntrjs amount of classical trajectories
	//initialize the breakup events
	int ibr=0;
	//first initialize the electron/positron
    float y[12],dydt[12];
		
    y[0]=bimpact;
    y[1]=0.;
	y[2]=0.;
	y[3]=0.;
	y[4]=zinit;
	y[5]=vinit;
	
	
	for(int i=0;i<1;i++)
	{
		//generate 5 random numbers in the uniform grid (0,1]
		//For micro canonical ensemble
		//printf("idx,i %i \t %i \n",idx,i);
		float ran0 = curand_uniform(&localState);
		float ran1 = curand_uniform(&localState);
		float ran2 = curand_uniform(&localState);
		float ran3 = curand_uniform(&localState);
		float ran4 = curand_uniform(&localState);
						
		
		float beta=ran0;
		float alpha=ran1*2.*pi;
		float psi=ran2*2.*pi;
		float cothe=2.*ran3-1.;
		float phi=ran4*2.*pi;
		
		//eccentricity of the orbit
		float ecc=sqrtf(1.-beta);
		//root-search for eccentric anomaly
		
		float xip=rootxi(alpha,ecc,0.0f,2.*pi);		
		float sixi=sinf(xip),coxi=cosf(xip);
		
		//initial unrotated coordinates and velocities
		float xr[3],vr[3];
		
		float den=tred*(1.0f-ecc*coxi);
		
		xr[0]=asem*sqrtf(beta)*sixi;
		xr[1]=0.0f;
		xr[2]=asem*(coxi-ecc);
		
		
		vr[0]=asem*sqrtf(beta)*coxi/den;
		vr[1]=0.;
		vr[2]=-asem*sixi/den;
		// check the energy
		float ekin0=.5*(vr[0]*vr[0]+vr[1]*vr[1]+vr[2]*vr[2]);
		float erel=ekin0-1.0f/sqrtf(xr[0]*xr[0]+xr[1]*xr[1]+xr[2]*xr[2]);
		
		//printf("ehyd,erel: %f \t  %f \n", ehyd,erel);
		//prepare Euler matrix
		float eul[3][3];
		float co1=cos(psi),si1=sin(psi),co3=cos(phi),si3=sin(phi);
		float sithe=sqrt(1.-cothe*cothe);
		
		eul[0][0]=co1*co3-si1*cothe*si3;
		eul[0][1]=-co1*si3-si1*cothe*co3;
		eul[0][2]=si1*sithe;
		eul[1][0]=si1*co3+co1*cothe*si3;
		eul[1][1]=-si1*si3+co1*cothe*co3;
		eul[1][2]=-co1*sithe;
		eul[2][0]=sithe*si3;
		eul[2][1]=sithe*co3;
		eul[2][2]=cothe;
		
		//rotate relative coordinates and velocities
		float va[3],xa[3];
		for(int ii=0;ii<3;ii++)
		{
			va[ii]=0.;
			xa[ii]=0.;
			for(int jj=0;jj<3;jj++)
			{
				va[ii]=va[ii]+eul[ii][jj]*vr[jj];
				xa[ii]=xa[ii]+eul[ii][jj]*xr[jj];
			}
			//transform to the lab-frame
			xa[ii]=xa[ii];
			va[ii]=va[ii];
		}
		
		//initialize the electron-positron
		
		
		y[6]=xa[0];
		y[7]=va[0];
		y[8]=xa[1];
		y[9]=va[1];
		y[10]=xa[2];
		y[11]=va[2];
		
		// check the initial energy
		float R1=sqrt(y[0]*y[0]+y[2]*y[2]+y[4]*y[4]+.2);
	    float R2=sqrt(y[6]*y[6]+y[8]*y[8]+y[10]*y[10]+.2);
		float dx=y[0]-y[6];
	    float dy=y[2]-y[8];
	    float dz=y[4]-y[10];	  
	    float R3=sqrt(dx*dx+dy*dy+dz*dz+.2);
		float ekin=.5*(y[1]*y[1]+y[3]*y[3]+y[5]*y[5]);
		float pkin=.5*(y[7]*y[7]+y[9]*y[9]+y[11]*y[11]);
		
		float en0=ekin+pkin-1./R1-1./R2+1./R3;
		
		derivs(0.,y,dydt);
		float run_time=0.;
		float tstep=1.;
		int iter=tmax/tstep;
		
		//Runge-Kutta Stepper
		float dym[12],dyt[12],yt[12],yout[12];
		float h=tstep,hh=h/2.,h6=h/6.;
		
		for(int it=0;it<iter;it++)
		{
			
			//propagate y,dydt a tstep
			//------------------------------------------------//
			for(int ij=0;ij<12;ij++)yt[ij]=y[ij]+hh*dydt[ij];
			derivs(run_time+hh,yt,dyt);
			for(int ik=0;ik<12;ik++)yt[ik]=y[ik]+hh*dyt[ik];
			derivs(run_time+hh,yt,dym);
			for (int il=0;il<12;il++)
				{
					yt[il]=y[il]+h*dym[il];
					dym[il] += dyt[il];
				}
			derivs(run_time+h,yt,dyt);
			for (int ir=0;ir<12;ir++)
				{
					yout[ir]=y[ir]+h6*(dydt[ir]+dyt[ir]+2.0*dym[ir]);
				
				}
		    //-----------------------------------------------//
			
			//re-initialize the stepper
			for(int iw=0;iw<12;iw++)y[iw]=yout[iw];
			derivs(run_time+h,y,dydt);
			//printf("%f\t%f\t%f\t%f\t%f\t%f\n",y[0],y[2],y[4],y[6],y[8],
			//y[10]);
			
			if(y[10]>=zasym)
				break;
			run_time=run_time+h;
			
		}
		
		//check the final energy
		
		R1=sqrt(y[0]*y[0]+y[2]*y[2]+y[4]*y[4]+.2);
	    R2=sqrt(y[6]*y[6]+y[8]*y[8]+y[10]*y[10]+.2);
		dx=y[0]-y[6];
	    dy=y[2]-y[8];
	    dz=y[4]-y[10];	  
	    R3=sqrt(dx*dx+dy*dy+dz*dz+.2);
		ekin=.5*(y[1]*y[1]+y[3]*y[3]+y[5]*y[5]);
		pkin=.5*(y[7]*y[7]+y[9]*y[9]+y[11]*y[11]);		
		float enf0=ekin+pkin-1./R1-1./R2+1./R3;
		//check the relative energy between incomping elecron and proton/antiproton
		float en1=ekin-1./R1;
		float en2=pkin-1./R2;
		
		if(en1>=0.0f && en2>=0.0f)
			ibr=ibr+1;
			
		//printf("bimpact,en1,en2,ibr: %f \t %f \t %f \t %i \n",
		//bimpact,en1,en2,ibr);
		
	}
	
	
	//copy local memory to global
	state[idx] = localState;
	a[idx]=ibr;
	//printf("here \n");
	//printf("inside: %i \n",ibr);
}



float cudaticsPs(float Ev)
{
	float pi=2.*asinf(1.0f);
	// number of threads and blocks
	// number of threads:range of impact parameters
	//number of blocks: shares the total number of classical trajectories required
	//int block_size,minGridSize;	
	//predefined number of trajectories for each block
	int ntrjs=5000;
	int block_size,minGridSize;
	
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &block_size, ticsKernel, 0, ntrjs); 
	//printf("block size, gridsize %i \t %i \n",block_size, minGridSize);
	block_size=32;
	minGridSize=1;
	int n_procs=minGridSize*block_size;
	//memory allocation in the host and device
	
	size_t size=n_procs *sizeof(int);
	int* a_h=(int*)malloc(size);
	int* a_d;
	cudaMalloc((void **) &a_d, size);
	//random_states
	curandState *devStates;
	cudaMalloc((void **) &devStates, n_procs *sizeof(curandState));
	int* ibr=(int*)malloc(size);
	
	for(int i=0;i<n_procs;i++)ibr[i]=0;
		
	
	

	
	// each thread gets a single impact parameters
	float bmax=18.;
	float bstep=bmax/(float)n_procs;
	
	//initialize the random numbers
	int s=12345;//seed
	rng_init<<<minGridSize,block_size>>>(devStates, s, n_procs);
	
	//pass it to parallel processing
	//each parallel unit counts the number of break-up events
	//run the kernel in a loop
	
	for(int iker=0;iker<ntrjs;iker++)
	{
		ticsKernel<<<minGridSize,block_size>>>(devStates,a_d,Ev,bstep);
		cudaDeviceSynchronize();
	
		cudaMemcpy(a_h,a_d, sizeof(int)*n_procs,cudaMemcpyDeviceToHost);
		
		for(int i=0;i<minGridSize;i++)
		{
			for(int j=0;j<block_size;j++)
			{
				int idx=i*block_size+j;
				ibr[idx] = ibr[idx]+a_h[idx];
			}
		
		}
	}
	
	// compute the cross section
	float tics=0.;
	for(int i=0;i<minGridSize;i++)
		{
			for(int j=0;j<block_size;j++)
			{
				int idx=i*block_size+j;
				float bimpact=bstep*(float)idx;
				//printf("b,ibr:%f \t %i\n",bimpact,ibr[idx]);
				if(ibr[idx]<=5)
					ibr[idx]=0;
				tics=tics+2.*pi*bstep*bimpact*(float)ibr[idx]/float(ntrjs);
			}
		
		}							
	//now free-up the space
	free(a_h);
	cudaFree(a_d);
	
	return tics;
}

int main()
{
	fstream pstics;
	pstics.open("ticscu.txt",ios::out);
	auto start = high_resolution_clock::now();
	// initial Positronium n,l states
	int nps=3,lps=0;
	//Threshold energy for ionization (a.u)
	float eth=.5/powf((float)nps,2);
	//initial Positronium Center-of-mass energy
	float Ev;
	for(int i=0;i<50;i++)
	{
		Ev=eth*27.21+(float)i*2.;
		float tics=cudaticsPs(Ev);
		printf("Break-up cross section at %f eV is: %f \n ",Ev,tics);
		pstics<<Ev<<"\t"<<tics<<endl;
		
	}
	
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<seconds>(stop - start);		
	cout<<"Duration (s)"<<"\t"<<duration.count()<<endl;
}