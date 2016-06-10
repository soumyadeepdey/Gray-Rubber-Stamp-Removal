

//~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*
// Program Name :      Same color Stamp Removal(same_color_stamp.cpp)
// 
// Project :  		DRD
// Author : 		Soumyadeep Dey
// Creation Date : 	JUNE  -2015.  Rights Reserved
//~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*

 
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <sys/stat.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv/cv.h"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"




using namespace cv;
using namespace std;

#include "xml/xml_Lib_src/xmlParser.h"
#include "xml/xml_Lib_src/schemaParser.cpp"
#include "xml/xml_Lib_src/xmlParser.cpp"
#include "xml/xml.h"
#include "xml/xml.cpp"

IITKGP_XML_IO::XML_IO IITKGP;

/*-------------------------------------------------------- Structure Declarations--------------------------------------------------------*/

typedef struct gapping
{
	int y1;
	int y2;
	int size;
	struct gapping *nxt;
}gp;

typedef struct imagestructure
{
	int x;
	int y;
	int label;
	int value;
	int mapped;
}is;


typedef struct connected_component
{
	struct imagestructure im;
	struct connected_component *nxt;
}cc;



typedef struct numberofcc
{
	struct connected_component *start;
	struct connected_component *last;
	float gray_hist[256];
	int number_of_component;
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	int centroidx;
	int centroidy;
	float histogram[256];
	int mean;
	double var;
	double std_dev;
	double skew;
	void calc_hist()
	{
	  
	  for(int i=0;i<256;i++)
	    histogram[i]=gray_hist[i]/number_of_component;
	}
	
	void calc_mean()
	{
	  mean = 0;
	  for(int i=0;i<256;i++)
	  {
	    mean = mean + (gray_hist[i]*i);
	  }
	  mean = mean/number_of_component;
	}
	
	
	void calc_dev()
	{
	  calc_mean();
	  int temp;
	  std_dev = 0.0;
	  var = 0.0;
	  skew = 0.0;
	  for(int i=0;i<256;i++)
	  {
	    for(int j=0;j<gray_hist[i];j++)
	    {
	      temp = i - mean;
	      var = var + (temp * temp);
	      skew = skew + (temp * temp *temp);
	    }
	  }
	  
	  var = var / number_of_component;
	  std_dev = sqrt(var);
	  
	}
	
	void calc_distribution()
	{
	  calc_hist();
	  calc_mean();
	  calc_dev();
	}
	
}nocc;


/*-----------------------------------------------------------------------------------------------------------------*/

/*------------------------------------------------------- Global variables----------------------------------------------------------------------*/


is **tempstructuredimage;

nocc *component;
nocc *originalcomponent;



Mat src, src_gray, binary_dst, output_image, erode_dst, dilate_dst, color_para, para_fill,dst;




int row,col,maximum;
int *img;
int *colorimage;
int *binary_image;
int *(*imgg);
int *(*imgg1);
int ncc;
int tncc;// total number of connected component
int ncco;//number of connected component in original image



//parameters
int old_doc=0;// variable to told doc old or new   // put 0 if not old(if you dont want to dilate the binary image before finding connected component)
int graphicsthreashold=2500; // IT IS THE THRESHOLD SET BASED ON 300 DPI SCAN TO SEPARATE GRAPHICS BASED ON CC SIZE AFTER STUDING ON 50 IMAGES

// parameters for binarization

int threshold_value = 0;
int threshold_type = 0;;
int const maximum_value = 255;
int const maximum_type = 4;
int const maximum_BINARY_value = 255;
int const blockSize=101;



Mat temp,hsv,open_dst,YCrCb;

float h_unitvec,s_unitvec,v_unitvec;

// varible hold all hsv values separately
vector<Mat> hsv_planes;
vector<Mat> YCrCb_planes;




int no_of_foregrnd_pix;

RNG rng(12345);

// substring name before .

char *substring;

/*-----------------------------------------------------------------------------------------------------------------*/

/*----------------------------------------------------------Function Declarations----------------------------------------------------------------------------------*/




void connectedcomponent(int **image);
int cclabeling(void);
void labelingdfs(int *i,int *j);






/*-------------------------------------------------------------------CONNECTED COMPONENT--------------------------------------------------------------------------------*/


void connectedcomponent(int **image)
{
	int i,j,label=1,*mapping,k;
	ncc=0;
	
	is *istemp;






	tempstructuredimage=(is **)malloc(row * sizeof(is *));
	if(tempstructuredimage == NULL)
	{
	  printf("can not allocate space\n");
	  exit(0);
	}
	for(i=0;i<row;i++)
	{
		tempstructuredimage[i]=(is *)malloc(col * sizeof(is));
		if(tempstructuredimage[i] == NULL)
		{
		  printf("can not allocate space\n");
		  exit(0);
		}
	}
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			tempstructuredimage[i][j].x=i;
			tempstructuredimage[i][j].y=j;
			tempstructuredimage[i][j].label=0;
			tempstructuredimage[i][j].value=image[i][j];
		}
	}



// LABELING BY DFS
	printf("before labeling\n");
	int noccbyla;
	noccbyla=(1-cclabeling());
	printf("after labeling\n");



// LABELING IS PERFECT

	mapping=NULL;
	k=0;
	int *tmp,count=1;
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(tempstructuredimage[i][j].label!=0)
			{
			 
				if(mapping!=NULL)
				{
					for(k=0;k<count-1;k++)
					{
						if(mapping[k]==tempstructuredimage[i][j].label)
						{
							tempstructuredimage[i][j].mapped=k;
							break;
						}		
					}
					if(k==count-1)
					{
						tmp=(int*)realloc(mapping,count*sizeof(int));
						if(tmp!=NULL)
						{
							mapping=tmp;
							mapping[count-1]=tempstructuredimage[i][j].label;
							tempstructuredimage[i][j].mapped=(count-1);
							count++;
						}
						else
						{
							printf("\nERROR IN REALLOCATING MAPPING ARREY\n");
							exit(1);
						}
					}// end of k==count
				}// end of if mapping !=null
				else
				{
				  
					tmp=(int*)realloc(mapping,count*sizeof(int));
					mapping=tmp;
					mapping[count-1]=tempstructuredimage[i][j].label;
					tempstructuredimage[i][j].mapped=(count-1);
					count++;
					
				}
				
			}// end of tempstructuredimage[i][j].label!=0
		}
	}// end of image

	

// MAPPING IS PERFECR TILL NOW

printf("MAPPING IS PERFECR TILL NOW\n");

	tncc=count-1;
	
	ncco = tncc;

// CREATING ARREY OF STRUCTURE POINTER  and help them to uniquely mapped

	cc *cctemp,*ccstart=NULL,*temp1;
	
	component=(nocc *)malloc((count-1)* sizeof(nocc));
	
	if(component == NULL)
	{
	  printf("memory can not be allocated \n");
	  exit (0);
	}

	for(i=0;i<(count-1);i++)
	{
		component[i].start=NULL;
		component[i].number_of_component=0;
		component[i].last=NULL;
		for(j=0;j<256;j++)
		  component[i].gray_hist[j]=0.0;
	}
	

	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(tempstructuredimage[i][j].label!=0)
			{
				
				if(component[tempstructuredimage[i][j].mapped].start==NULL)
				{
					
					if(tempstructuredimage[i][j].mapped<0||tempstructuredimage[i][j].mapped>=count-1)
						{
							printf("error\n");
							printf("%d\t%d\t%d\t%d",tempstructuredimage[i][j].mapped,tempstructuredimage[i][j].x,tempstructuredimage[i][j].y,tempstructuredimage[i][j].label);
							exit(1);
						}
				
					
					cctemp=(cc *)malloc(sizeof(cc));
					if(cctemp == NULL)
					{
					  printf("memory can not be allocated \n");
					  exit (0);
					}
					cctemp->im.x=i;
					cctemp->im.y=j;
					cctemp->im.label=tempstructuredimage[i][j].label;
					cctemp->im.mapped=tempstructuredimage[i][j].mapped;
					cctemp->im.value=src_gray.data[i*col+j];
					cctemp->nxt=NULL;
					ccstart=(cc *)malloc(sizeof(cc));
					ccstart=cctemp;
					component[tempstructuredimage[i][j].mapped].start=cctemp;
					component[tempstructuredimage[i][j].mapped].last=cctemp;
					component[tempstructuredimage[i][j].mapped].number_of_component=1;
					component[tempstructuredimage[i][j].mapped].xmin=i;
					component[tempstructuredimage[i][j].mapped].ymin=j;
					component[tempstructuredimage[i][j].mapped].xmax=i;
					component[tempstructuredimage[i][j].mapped].ymax=j;
					component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]=component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]+1;
					
				}//end of if  i.e. first component of the connected component
				else
				{
					
					cctemp=(cc *)malloc(sizeof(cc));
					if(cctemp == NULL)
					{
					  printf("memory can not be allocated \n");
					  exit (0);
					}
					cctemp->im.x=i;
					cctemp->im.y=j;
					cctemp->im.label=tempstructuredimage[i][j].label;
					cctemp->im.mapped=tempstructuredimage[i][j].mapped;
					cctemp->im.value=src_gray.data[i*col+j];
					cctemp->nxt=NULL;
					if(component[tempstructuredimage[i][j].mapped].last->nxt==NULL)
						component[tempstructuredimage[i][j].mapped].last->nxt=cctemp;
					else
						printf("ERROR\n");
					component[tempstructuredimage[i][j].mapped].last=cctemp;
					component[tempstructuredimage[i][j].mapped].number_of_component=(component[tempstructuredimage[i][j].mapped].number_of_component)+1;
					if(component[tempstructuredimage[i][j].mapped].xmin>i)
						component[tempstructuredimage[i][j].mapped].xmin=i;
					if(component[tempstructuredimage[i][j].mapped].ymin>j)
						component[tempstructuredimage[i][j].mapped].ymin=j;
					if(component[tempstructuredimage[i][j].mapped].xmax<i)
						component[tempstructuredimage[i][j].mapped].xmax=i;
					if(component[tempstructuredimage[i][j].mapped].ymax<j)
						component[tempstructuredimage[i][j].mapped].ymax=j;
					
					component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]=component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]+1;

				}// end of else i.e. not 1st component of connected component
			}// end of if label

			
	
		}// end of j

	
	
	}// end of i
	printf("CC done\n");
	free(mapping);

}


// LABELING WITH DFS

int cclabeling(void)
{
	int label=1;
	int i,j,k,l;
        
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
	
			if(tempstructuredimage[i][j].label==0&&tempstructuredimage[i][j].value==0)
			{
			
				tempstructuredimage[i][j].label=label;
				labelingdfs(&i,&j);
				label=label+1;
				
		
			}
			
			
		}
	}
	return(label);
	
}

void labelingdfs(int *k,int *l)
{
	int i,j,m,n;
	i=*k;
	j=*l;


	


// 8 NEIGHBOURS

/*
	int n1x,n2x,n3x,n4x,n5x,n6x,n7x,n8x;
	int n1y,n2y,n3y,n4y,n5y,n6y,n7y,n8y;
	
	n1x=i-1;
	n1y=j-1;
	n2x=i-1;
	n2y=j;
	n3x=i-1;
	n3y=j+1;
	n4x=i;
	n4y=j-1;
	n5x=i;
	n5y=j+1;
	n6x=i+1;
	n6y=j-1;
	n7x=i+1;
	n7y=j;
	n8x=i+1;
	n8y=j+1;

*/

	
	
if(tempstructuredimage[i][j].value==0&&tempstructuredimage[i][j].label!=0)
{



	if(i-1>=0&&j-1>=0)
	{
		if(tempstructuredimage[i-1][j-1].value==0&&tempstructuredimage[i-1][j-1].label==0)
		{
			//printf("N1\n");
			tempstructuredimage[i-1][j-1].label=tempstructuredimage[i][j].label;
			m=i-1;
			n=j-1;
			labelingdfs(&m,&n);
		}
	}
	if(i-1>=0)
	{
		if(tempstructuredimage[i-1][j].value==0&&tempstructuredimage[i-1][j].label==0)
		{
			//printf("N2\n");
			tempstructuredimage[i-1][j].label=tempstructuredimage[i][j].label;
			m=i-1;
			n=j;
			labelingdfs(&m,&n);
		}
	}
	if(i-1>=0&&j+1<col)
	{
		if(tempstructuredimage[i-1][j+1].value==0&&tempstructuredimage[i-1][j+1].label==0)
		{
			//printf("N3\n");
			tempstructuredimage[i-1][j+1].label=tempstructuredimage[i][j].label;
			m=i-1;
			n=j+1;
			labelingdfs(&m,&n);
		}
	}
	if(j-1>=0)
	{
		if(tempstructuredimage[i][j-1].value==0&&tempstructuredimage[i][j-1].label==0)
		{
			//printf("N4\n");
			tempstructuredimage[i][j-1].label=tempstructuredimage[i][j].label;
			m=i;
			n=j-1;
			labelingdfs(&m,&n);
		}
	}
	if(j+1<col)
	{
		if(tempstructuredimage[i][j+1].value==0&&tempstructuredimage[i][j+1].label==0)
		{
			//printf("N5\n");
			tempstructuredimage[i][j+1].label=tempstructuredimage[i][j].label;
			m=i;
			n=j+1;
			labelingdfs(&m,&n);
		}
	}
	if(i+1<row&&j-1>=0)
	{
		if(tempstructuredimage[i+1][j-1].value==0&&tempstructuredimage[i+1][j-1].label==0)
		{
			//printf("N6\n");
			tempstructuredimage[i+1][j-1].label=tempstructuredimage[i][j].label;
			m=i+1;
			n=j-1;
			labelingdfs(&m,&n);
		}
	}
	if(i+1<row)
	{
		if(tempstructuredimage[i+1][j].value==0&&tempstructuredimage[i+1][j].label==0)
		{
			//printf("N7\n");
			tempstructuredimage[i+1][j].label=tempstructuredimage[i][j].label;
			m=i+1;
			n=j;
			labelingdfs(&m,&n);
		}
	}
	if(i+1<row&&j+1<col)
	{
		if(tempstructuredimage[i+1][j+1].value==0&&tempstructuredimage[i+1][j+1].label==0)
		{
			//printf("N8\n");
			tempstructuredimage[i+1][j+1].label=tempstructuredimage[i][j].label;
			m=i+1;
			n=j+1;
			labelingdfs(&m,&n);
		}

	}

}
	
}

/*---------------------------------------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------Binarization-------------------------------------------*/


void OtsuBinarization( )
{
   
 int r,c;
 r = src.rows;
 c = src.cols;
 const int L = 256;
 float hist[256]={0.0};
 int x,y,N;
 int graylevel; int i,k;
 float ut = 0.0;

 int max_k;
 int max_sigma_k;
 float wk;
 float uk;
 float sigma_k;
 //calculate grayscale histogram
 for ( x=0; x < r; ++x)
 for( y=0; y < c; ++y)
 {

 graylevel = src_gray.data[x*c+y];
 hist[graylevel]+=1.0;
 }
 
 N = r*c;
 
 //normalize histogram
 for ( i=0; i<L ;i++) 
 hist[i]/=N;
 
 
 
 for ( i=0; i< L ;i++)
 ut+=i*hist[i];
 
  max_k=0;
  max_sigma_k=0;
 for ( k=0; k < L;++k)
 {
    wk = 0.0 ;
   for ( i = 0; i <=k;++i)
    wk += hist[i];
    uk = 0.0;
   for ( i = 0; i <=k;++i)
    uk+= i*hist[i];
 
   sigma_k = 0.0;
   if (wk !=0.0 && wk!=1.0)
   sigma_k  = ((ut*wk - uk)*(ut*wk - uk))/(wk*(1-wk));
 
   if (sigma_k > max_sigma_k)
    {
     max_k = k;
     max_sigma_k = sigma_k;
    }
 } /*main for*/
 
 src_gray.copyTo(binary_dst);
 
 for ( x=0; x < r; ++x )
 for ( y=0; y < c; ++y )
  {
  
  graylevel = src_gray.data[x*c+y];
  if (graylevel < max_k)
  binary_dst.data[x*c+y] = 0;
   else
   binary_dst.data[x*c+y] = 255;
  }
  
  
}


void *binarization()
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

	/// Convert the image to Gray
  	
	
	cvtColor(src, src_gray, CV_RGB2GRAY);
	
	//adaptiveThreshold(  src_gray, binary_dst, maximum_BINARY_value, ADAPTIVE_THRESH_GAUSSIAN_C,  threshold_type,  blockSize, 20);
	
	//threshold( src_gray, global_binary_dst, threshold_value, maximum_BINARY_value, 3);
	
	
	OtsuBinarization();
	//global_threshold();

}



/*-------------------------------------------------MAKE DIRECTORY FUNCTION-------------------------------------------*/



/*------------------------------------------------------------------------------------------------------------------------------------------------*/



void makedir(char *name)
{
	int status;
	status=mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}



/*-------------------------------------------------------------------------------------------------------------------------*/



/*-------------------------------------------------------EROTION WITH 4 NEIGHBOURHOOD-------------------------------------------------------------*/

Mat erosion(Mat image)
{
	
	int i,j;
	Mat tempimage;
	image.copyTo(tempimage);
	for(i=0;i<row;i++)
	{
	  for(j=0;j<col;j++)
	    tempimage.data[i*col+j] = 255;
	}
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(image.data[i*col+j]==0)
			{
				if(i-1<0||i+1>=row||j-1<0||j+1>=col)
					tempimage.data[i*col+j]=255;
				else if(image.data[(i-1)*col+j]==0&&image.data[(i+1)*col+j]==0&&image.data[i*col+(j-1)]==0&&image.data[i*col+(j+1)]==0)
					tempimage.data[i*col+j]=0;
				else
					tempimage.data[i*col+j]=255;
			}
			else
				tempimage.data[i*col+j]=255;
		}
	}

	return (tempimage);
	
		
}

/*-------------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------- BOUNDARY EXTRACTION--------------------------------------------------*/

Mat boundaryextraction(Mat image)
{
	
	int i,j,k;

	Mat erodedimage;
	Mat extractedimage;
	
	image.copyTo(erodedimage);
	image.copyTo(extractedimage);
	
	for(i=0;i<row;i++)
	{
	  for(j=0;j<col;j++)
	  {
	    erodedimage.data[i*col+j] = 255;
	    extractedimage.data[i*col+j] = 255;
	  }
	}
	
	erodedimage=erosion(image);
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(image.data[i*col+j]==erodedimage.data[i*col+j])
				extractedimage.data[i*col+j]=255;
			else
				extractedimage.data[i*col+j]=0;
		}
	}

	return(extractedimage);
	
	
}

/*-------------------------------------------------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------------------------------------------------------------------------*/


/*---------------------------------------------------------------------distribution--------------------------------------------------------------*/



void distribution()
{
  int i,j,k;
  
 // printf("\t hue \t sat \t val \t h_s \t s_v \t v_h \t h_s_v \n");
  FILE *h,*s,*v,*h_s,*s_v,*v_h,*h_s_v,*all, *prob, *hsv_3d, *gray;
  char *name;
  
  
  makedir("distribution");

   h = fopen("distribution/hue.txt","w");

   s = fopen("distribution/sat.txt","w");

   v = fopen("distribution/val.txt","w");

   h_s = fopen("distribution/hue_sat.txt","w");
   
   s_v = fopen("distribution/sat_val.txt","w");

   v_h = fopen("distribution/val_hue.txt","w");

   h_s_v = fopen("distribution/hue_sat_val.txt","w");

  all = fopen("distribution/all.txt","w");

  prob = fopen("distribution/probability.txt","w");

  hsv_3d = fopen("distribution/3d_hsv.txt","w");

  FILE* sv_jd = fopen("distribution/sv_jd.txt","w");

  FILE* hsv_jd = fopen("distribution/hsv_jd.txt","w");
  
  gray = fopen("distribution/gray_dist.txt","w");
  
  
  
   
  
  
  
  fprintf(h,"\n");
  fprintf(s,"\n");
  fprintf(v,"\n");
  fprintf(h_s,"\n");
  fprintf(s_v,"\n");
  fprintf(v_h,"\n");
  fprintf(h_s_v,"\n");
  fprintf(all,"\n");
  fprintf(prob,"\n");
  fprintf(hsv_3d,"\n");
  fprintf(sv_jd,"\n");
  fprintf(hsv_jd,"\n");
  fprintf(gray,"\n");
  
  
  Mat hs,sv,vh,hue_sat_val;
  
  binary_dst.copyTo(hs);
  binary_dst.copyTo(sv);
  binary_dst.copyTo(vh);
  binary_dst.copyTo(hue_sat_val);
  
  int tottalnz = 0;
  
  float *ph,*ps,*pv,*ph_s,*ps_v,*pv_h,*ph_s_v,*pgray;
  
  ph = (float *) malloc ( 256 * sizeof(float));
  ps = (float *) malloc ( 256 * sizeof(float));
  pv = (float *) malloc ( 256 * sizeof(float));
  pgray = (float *) malloc ( 256 * sizeof(float));
  ph_s = (float *) malloc ( (256 + 256) * sizeof(float));
  ps_v = (float *) malloc ( (256 + 256) * sizeof(float));
  pv_h = (float *) malloc ( (256 + 256) * sizeof(float));
  ph_s_v = (float *) malloc ( (256 + 256 + 256) * sizeof(float));
  
  float **p_sh,**p_vh, **psv_h;
  
  
  p_sh = (float **) malloc ( 256 * sizeof(float*));
  p_vh = (float **) malloc ( 256 * sizeof(float*));
  psv_h = (float **) malloc ( 256 * sizeof(float*));
  
  for(i=0;i<256;i++)
  {
    p_sh[i] = (float *) malloc ( 256 * sizeof(float));
    p_vh[i] = (float *) malloc ( 256 * sizeof(float));
    psv_h[i] = (float *) malloc ( 256 * 2 * sizeof(float));
  }

  float ***p_vsh;
  
  p_vsh = (float ***) malloc ( 256 * sizeof(float**));
  
  for(i=0;i<256;i++)
  {
    p_vsh[i] = (float **) malloc ( 256 * sizeof(float*));
  }
  for(i=0;i<256;i++)
  {
    for(j=0;j<256;j++)
      p_vsh[i][j] = (float *) malloc ( 256 * sizeof(float));
  }
  
  
  float *p_weighted;
  
  p_weighted = (float *) malloc (5 * 256 *sizeof(float));
  
  
  for(i=0;i<=255;i++)
  {
    ph[i]=0.0;
    ps[i]=0.0;
    pv[i]=0.0;
    pgray[i]=0.0;
    for(j=0;j<256;j++)
    {
      p_sh[i][j]=0.0;
      p_vh[i][j]=0.0;
      for(k=0;k<256;k++)
	p_vsh[i][j][k]=0.0;
    }
    for(j=0;j<256*2;j++)
    {
      psv_h[i][j]=0.0;
    }
 
  }
  
  for(i=0;i<=510;i++)
  {
    ph_s[i] = 0.0;
    ps_v[i] = 0.0;
    pv_h[i] = 0.0;
  }
  
  for(i=0;i<=765;i++)
    ph_s_v[i] = 0.0;
  
  for(i=0;i<5*256;i++)
    p_weighted[i] = 0.0;
  
  
  int temp1,temp2,temp3,temp4,temp5;

   
  float mean_hsv=0.0;
  double stddev_hsv=0.0;
  
  for(i = 0; i < src.rows; i++)
  {
    for(j = 0; j < src.cols; j++)
    {
      if(binary_dst.data[i*src.cols+j] != 255)
      {
	
	ph[hsv_planes[0].data[i*src.cols+j]] = ph[hsv_planes[0].data[i*src.cols+j]] +1;
	ps[hsv_planes[1].data[i*src.cols+j]] = ps[hsv_planes[1].data[i*src.cols+j]] +1;
	pv[hsv_planes[2].data[i*src.cols+j]] = pv[hsv_planes[2].data[i*src.cols+j]] +1;
	pgray[src_gray.data[i*src.cols+j]] = pgray[src_gray.data[i*src.cols+j]] +1;
	
	temp1 = hsv_planes[0].data[i*src.cols+j] + hsv_planes[1].data[i*src.cols+j];
	temp2 = hsv_planes[1].data[i*src.cols+j] + hsv_planes[2].data[i*src.cols+j];
	temp3 = hsv_planes[2].data[i*src.cols+j] + hsv_planes[0].data[i*src.cols+j];
	temp4 = hsv_planes[0].data[i*src.cols+j] + hsv_planes[1].data[i*src.cols+j] + hsv_planes[2].data[i*src.cols+j];
	temp5 = (h_unitvec * hsv_planes[0].data[i*src.cols+j]) + (s_unitvec * hsv_planes[1].data[i*src.cols+j]) + (v_unitvec * hsv_planes[2].data[i*src.cols+j]);
	
	ph_s[temp1] = ph_s[temp1] + 1;
	ps_v[temp2] = ps_v[temp2] + 1;
	pv_h[temp3] = pv_h[temp3] + 1;
	ph_s_v[temp4] = ph_s_v[temp4] + 1;
	p_weighted[temp5] = p_weighted[temp5] + 1;
	
	mean_hsv = mean_hsv + temp4;
	
	
	//chk distribution
	
	p_sh[hsv_planes[0].data[i*src.cols+j]][hsv_planes[1].data[i*src.cols+j]] = p_sh[hsv_planes[0].data[i*src.cols+j]][hsv_planes[1].data[i*src.cols+j]] +1;
	p_vh[hsv_planes[0].data[i*src.cols+j]][hsv_planes[2].data[i*src.cols+j]] = p_vh[hsv_planes[0].data[i*src.cols+j]][hsv_planes[2].data[i*src.cols+j]] +1;
	
	
	p_vsh[hsv_planes[0].data[i*src.cols+j]][hsv_planes[1].data[i*src.cols+j]][hsv_planes[2].data[i*src.cols+j]] = p_vsh[hsv_planes[0].data[i*src.cols+j]][hsv_planes[1].data[i*src.cols+j]][hsv_planes[2].data[i*src.cols+j]] + 1;
	
	psv_h[hsv_planes[0].data[i*src.cols+j]][hsv_planes[1].data[i*src.cols+j]+hsv_planes[2].data[i*src.cols+j]] = psv_h[hsv_planes[0].data[i*src.cols+j]][hsv_planes[1].data[i*src.cols+j]+hsv_planes[2].data[i*src.cols+j]] + 1;
	
	tottalnz++;
	
      }
    }
  }

 mean_hsv = mean_hsv/tottalnz; 
 
 for(i = 0; i < src.rows; i++)
  {
    for(j = 0; j < src.cols; j++)
    {
      if(binary_dst.data[i*src.cols+j] != 255)
      {
	temp4 = hsv_planes[0].data[i*src.cols+j] + hsv_planes[1].data[i*src.cols+j] + hsv_planes[2].data[i*src.cols+j] - mean_hsv;
	stddev_hsv = stddev_hsv + (temp4 * temp4);
      }
    }
  }
  
  stddev_hsv = stddev_hsv/tottalnz;
  stddev_hsv = sqrt(stddev_hsv);
  
  printf("mean_hsv = %f\t stddev_hsv = %lf\n",mean_hsv,stddev_hsv);
  
  
/*  
  for(i=0;i<(5*256);i++)
  {
    if(i<=765)
    {
      if(i<=510)
      {
	if(i<=255)
	{
	  ph[i]=ph[i]/tottalnz;
	  fprintf(prob,"\t %f",ph[i]);
	  ps[i]=ps[i]/tottalnz;
	  fprintf(prob,"\t %f",ps[i]);
	  pv[i]=pv[i]/tottalnz;
	  fprintf(prob,"\t %f",pv[i]);
	  ph_s[i] = ph_s[i] / tottalnz;
	  fprintf(prob,"\t %f",ph_s[i]);
	  ps_v[i] = ps_v[i] / tottalnz;
	  fprintf(prob,"\t %f",ps_v[i]);
	  pv_h[i] = pv_h[i] / tottalnz;
	  fprintf(prob,"\t %f",pv_h[i]); 
	  ph_s_v[i] = ph_s_v[i] / tottalnz;
	  fprintf(prob,"\t %f\n",ph_s_v[i]);
	  p_weighted[i] = p_weighted[i] / tottalnz;
	  fprintf(prob,"\t %f\n",p_weighted[i]);
	}
	
	ph_s[i] = ph_s[i] / tottalnz;
	fprintf(prob,"\t \t \t \t %f",ph_s[i]);
	ps_v[i] = ps_v[i] / tottalnz;
	fprintf(prob,"\t %f",ps_v[i]);
	pv_h[i] = pv_h[i] / tottalnz;
	fprintf(prob,"\t %f",pv_h[i]);
	ph_s_v[i] = ph_s_v[i] / tottalnz;
	fprintf(prob,"\t %f\n",ph_s_v[i]);
	p_weighted[i] = p_weighted[i] / tottalnz;
	fprintf(prob,"\t %f\n",p_weighted[i]);
      }
     
      ph_s_v[i] = ph_s_v[i] / tottalnz;
      fprintf(prob,"\t \t \t \t \t \t \t %f\n",ph_s_v[i]);
      p_weighted[i] = p_weighted[i] / tottalnz;
      fprintf(prob,"\t %f\n",p_weighted[i]);
    }
    p_weighted[i] = p_weighted[i] / tottalnz;
    fprintf(prob,"\t \t \t \t \t \t \t \t %f\n",p_weighted[i]);
      
  }
  
  fclose(prob);*/
  
  
  for(i=0;i<256;i++)
  {
    ph[i]=ph[i]/tottalnz;
    ps[i]=ps[i]/tottalnz;
    pv[i]=pv[i]/tottalnz;
    pgray[i]=pgray[i]/tottalnz;
    fprintf(hsv_3d,"\t %f",ph[i]);
    fprintf(h,"\t %f\n",ph[i]);
    fprintf(hsv_3d,"\t %f",ps[i]);
    fprintf(s,"\t %f\n",ps[i]);
    fprintf(hsv_3d,"\t %f\n",pv[i]);
    fprintf(v,"\t %f\n",pv[i]);
    fprintf(gray,"\t %f\n",pgray[i]);
  }
  
  fclose(hsv_3d);
  
  
  for(i=0;i<=255;i++)
  {
    for(j=0;j<=255;j++)
    {
      p_sh[i][j] = p_sh[i][j]/tottalnz;
      p_vh[i][j] = p_vh[i][j]/tottalnz;
      for(k=0;k<=255;k++)
      {
	p_vsh[i][j][k] = p_vsh[i][j][k]/tottalnz;
      }
    }
  }
  
  
  for(i=0;i<=255;i++)
  {
    for(j=0;j<=255;j++)
    {
       fprintf(sv_jd,"\t %f",p_sh[i][j]);
       fprintf(sv_jd,"\t %f\n",p_vh[i][j]);
   
    }
  }
  
  for(i=0;i<=255;i++)
  {
    for(j=0;j<256*2;j++)
    {
      psv_h[i][j] = psv_h[i][j]/tottalnz;
      fprintf(hsv_jd,"\t %f\n",psv_h[i][j]);
    }
  }
  
  fclose(sv_jd);
  fclose(hsv_jd);
  
  
  fclose(h);
  fclose(s);
  fclose(v);
  fclose(h_s);
  fclose(s_v);
  fclose(v_h);
  fclose(h_s_v);
  
  
}



/*-------------------------------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------cut string upto( .)-------------------------------------------*/

void input_image_name_cut(char *s) 
{
                 
                     int i,j; 
		     
		     substring = (char *)malloc(501 * sizeof(char));
              
                 for(i=0; i <= strlen(s)-1; i++)
                      {
			
                       if (s[i]!='.' )
		        substring[i] = s[i];
		       else
			 break;
                       }
                       substring[i] = '\0';
                 
                
		     
		     
                      
      }


/*-------------------------------------------------------------------------------------------------------------------------------------------*/









int main(int argc, char *argv[])
{
	    
  if(argc!=3)
	{
	      printf("not proper argument type\n");
	      printf("Proper format:\n");
	      printf("./gray <input xml file name> <output xml file name>");
	      printf("for eg: \n ./gray input.xml output.xml");
	      exit (0);
	}
  
  
  int i,j,k,m;
	    
	   
	    char *output;
	char filename[501];
	char *name;

	char *input_image_name;
	
	input_image_name=IITKGP.readXML(argv[1]);

	src=imread(input_image_name,1);

	input_image_name_cut(input_image_name);
	    
	    output = (char *) malloc(501 * sizeof(char));
	    
	 strcpy(output,substring);
	

	    
	    no_of_foregrnd_pix = 0;
	    

	    
	    row = src.rows;
	    col = src.cols;
	    
	    binarization();
	    //imwrite("binary_image.png",binary_dst); 
	    
	   
	    
	    src.copyTo(temp);
	    
	    for(i=0;i<src.rows;i++)
	    {
	      for(j=0;j<src.cols;j++)
	      { 
		if(binary_dst.data[i*src.cols+j]==255)
		{
		  for(k=0;k<3;k++)
		  {
		    temp.data[(i*src.cols+j)*3+k]=255;
		  }
		}
		else
		  no_of_foregrnd_pix++; // foreground pixel
	      }
	    }
	    
	   
	   // imwrite("uniback.png",temp);
	    
	 Mat bextracted_image;
	 bextracted_image=boundaryextraction(binary_dst);
	 //imwrite("boundary_image.png",bextracted_image);
	 
	 Mat tempimg;
	 
	 binary_dst.copyTo(tempimg);
	 
	 for(i=0;i<src.rows*src.cols;i++)
	 {
	   if(bextracted_image.data[i]==0)
	     tempimg.data[i]=255;
	 }
	    
	// FINDING CONNECTED COMPONENT on the IMAGE

	int **im;
	im=(int **)malloc(row * sizeof(int *));
	
	if(im == NULL)
	{
	  printf("memory can not be allocated \n");
	  exit (0);
	}
	
	for(i=0;i<row;i++)
	{
		im[i]=(int *)malloc(col * sizeof(int));
		
		if(im[i] == NULL)
		{
		  printf("memory can not be allocated \n");
		  exit (0);
		}
	}
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
		  im[i][j]=binary_dst.data[i*col+j]; 
		 // im[i][j]=tempimg.data[i*col+j]; 
		}
	}



	connectedcomponent(im);	
	
	printf("number of cc = %d\n",ncco);
	
	originalcomponent=(nocc *)malloc(ncco* sizeof(nocc));

	
	if(originalcomponent == NULL)
	{
	  printf("memory can not be allocated \n");
	  exit (0);
	}
	
	int max_std_dev=0;
	
	for(i=0;i<ncco;i++)
	{
		originalcomponent[i].start=component[i].start;
		originalcomponent[i].last=component[i].last;
		originalcomponent[i].number_of_component=component[i].number_of_component;
		originalcomponent[i].xmin=component[i].xmin;
		originalcomponent[i].ymin=component[i].ymin;
		originalcomponent[i].xmax=component[i].xmax;
		originalcomponent[i].ymax=component[i].ymax;
		for(j=0;j<256;j++)
		  originalcomponent[i].gray_hist[j]=component[i].gray_hist[j];
		originalcomponent[i].calc_distribution();
		if(originalcomponent[i].std_dev > max_std_dev)
		  max_std_dev = originalcomponent[i].std_dev;
	}
	
	float std_dev_hist[max_std_dev+1];
	
	for(i=0;i<max_std_dev+1;i++)
	  std_dev_hist[i]=0.0;
	int mean_std_dev=0;
	int temp_var;
	for(i=0;i<ncco;i++)
	{
	  temp_var = originalcomponent[i].std_dev;
	  std_dev_hist[temp_var] = std_dev_hist[temp_var] +1;
	  mean_std_dev = mean_std_dev + originalcomponent[i].std_dev;
	}
	
	mean_std_dev = mean_std_dev/ncco;
	
	for(i=0;i<max_std_dev+1;i++)
	  std_dev_hist[i]=std_dev_hist[i]/ncco;
	
	//FILE *fp_std_hist;
	
	//fp_std_hist = fopen("std_dev_hist.txt","w");
	
	//for(i=0;i<max_std_dev+1;i++)
	 // fprintf(fp_std_hist,"\n\t%f\t",std_dev_hist[i]);
	//fclose(fp_std_hist);
	
	//FILE *fp;
	//FILE *fp_hist;
	//fp = fopen("mixed_dist.txt","w");
	//fp_hist = fopen("hist_data.txt","w");
	
	Mat tempo,stamp_removed;
	
	src.copyTo(tempo);
	binary_dst.copyTo(stamp_removed);
	cc *temp;
	int ic,jc;
	for(i=0;i<ncco;i++)
	{
	  //fprintf(fp,"\n\t%d\t%lf\t%lf",originalcomponent[i].mean,originalcomponent[i].var,originalcomponent[i].std_dev);
	  //fprintf(fp_hist,"\n");
 	  //for(j=0;j<256;j++)
 	  //{
	   // fprintf(fp_hist,"\t%f",originalcomponent[i].histogram[j]);
 	  //}
	  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	  rectangle(tempo, Point(originalcomponent[i].ymin,originalcomponent[i].xmin), Point(originalcomponent[i].ymax,originalcomponent[i].xmax), color, 2, 8, 0);
	  
	  if(originalcomponent[i].std_dev < 26.0)
	  {
	   // printf("hello %d\n",i);
	    temp=(cc *)malloc(sizeof(cc));
	    if(temp == NULL)
	    {
	      printf("memory can not be allocated \n");
	      exit (0); 
	    }
	    temp = originalcomponent[i].start;
	    while(temp!=NULL)
	    {
	      ic = temp->im.x;
	      jc = temp->im.y;
	      if(src_gray.data[ic*col+jc] > 65)
	      {
		tempo.data[(ic*col+jc)*3+0]=0;
		tempo.data[(ic*col+jc)*3+1]=0;
		tempo.data[(ic*col+jc)*3+2]=255;
		stamp_removed.data[ic*col+jc] = 255;
	      }
	      else
	      {
		tempo.data[(ic*col+jc)*3+0]=255;
		tempo.data[(ic*col+jc)*3+1]=0;
		tempo.data[(ic*col+jc)*3+2]=0;
	      }
	      temp = temp->nxt;
	    }
	  }
	  
	}
	
	//fclose(fp);
	//fclose(fp_hist);
	
	//imwrite("new_stamp-removed.png",stamp_removed);
	
	name = (char *) malloc ( 501 * sizeof(char));
	    strcpy(name,output);
	    strcat(name,"_stamp_removed.png");
	    
	
	//imwrite("New_image_test.png",tempo);
	imwrite(name,stamp_removed);

/*----------------------------------------------------------------------------------------------------------*/
  
	    /*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


	char *xmlchild;
	xmlchild = "Binarization";
	IITKGP.writemyxml(argv[1],argv[2],xmlchild,input_image_name,name,"NULL", "GrayStampRemoval",0,0,0,0); 
	    

	    
  
  return 0;
}
