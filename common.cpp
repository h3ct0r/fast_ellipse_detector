/*
This code is intended for academic use only.
You are free to use and modify the code, at your own risk.

If you use this code, or find it useful, please refer to the paper:

Michele Fornaciari, Andrea Prati, Rita Cucchiara,
A fast and effective ellipse detector for embedded vision applications
Pattern Recognition, Volume 47, Issue 11, November 2014, Pages 3693-3708, ISSN 0031-3203,
http://dx.doi.org/10.1016/j.patcog.2014.05.012.
(http://www.sciencedirect.com/science/article/pii/S0031320314001976)


The comments in the code refer to the abovementioned paper.
If you need further details about the code or the algorithm, please contact me at:

michele.fornaciari@unimore.it

last update: 23/12/2014
*/

#include "common.h"




void Canny2(InputArray image, OutputArray _edges,
			OutputArray _sobel_x, OutputArray _sobel_y,
                double threshold1, double threshold2,
                int apertureSize, bool L2gradient )
{
    Mat src = image.getMat();
    _edges.create(src.size(), CV_8U);
	_sobel_x.create(src.size(), CV_16S);
	_sobel_y.create(src.size(), CV_16S);

    // Use OpenCV 4's Canny implementation
    Sobel(src, _sobel_x, CV_16S, 1, 0, apertureSize);
    Sobel(src, _sobel_y, CV_16S, 0, 1, apertureSize);
    Canny(_sobel_x, _sobel_y, _edges, threshold1, threshold2, L2gradient);
};


void Labeling(Mat1b& image, vector<vector<Point> >& segments, int iMinLength)
{
	#define RG_STACK_SIZE 2048

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	int stack2[RG_STACK_SIZE];
	#define RG_PUSH2(a) (stack2[sp2] = (a) , sp2++)
	#define RG_POP2(a) (sp2-- , (a) = stack2[sp2])

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	Point stack3[RG_STACK_SIZE];
	#define RG_PUSH3(a) (stack3[sp3] = (a) , sp3++)
	#define RG_POP3(a) (sp3-- , (a) = stack3[sp3])

	int i,w,h, iDim;
	int x,y;
	int x2,y2;
	int sp2; // stack pointer
    int sp3;

	Mat_<uchar> src = image.clone();
	w = src.cols;
	h = src.rows;
	iDim = w*h;

	Point point;
	for (y=0; y<h; ++y)
	{
		for (x=0; x<w; ++x)
		{
			if ((src(y,x))!=0)   //punto non etichettato: seme trovato
			{
				// per ogni oggetto
				sp2 = 0;
				i = x + y*w;
				RG_PUSH2(i);

				// vuoto la lista dei punti
	    		sp3=0;
  		  		while (sp2>0)
				{// rg tradizionale

					RG_POP2(i);
					x2=i%w;
					y2=i/w;



					point.x=x2;
					point.y=y2;

					if(src(y2,x2))
					{
						RG_PUSH3(point);
						src(y2,x2) = 0;
					}
					
					// Inserisco i nuovi punti nello stack solo se esistono
					// e sono punti da etichettare

					// 4 connessi
					// sx
					if (x2>0 &&   (src(y2, x2-1)!=0))
						RG_PUSH2(i-1);
					// sotto
					if (y2>0 &&   (src(y2-1, x2)!=0))
						RG_PUSH2(i-w);
					// sopra
					if (y2<h-1 &&   (src(y2+1, x2)!=0))
						RG_PUSH2(i+w);
					// dx
					if (x2<w-1 &&   (src(y2, x2+1)!=0))
						RG_PUSH2(i+1);

					// 8 connessi
					if (x2>0 && y2>0 &&   (src(y2-1,x2-1)!=0))
						RG_PUSH2(i-w-1);
					if (x2>0 && y2<h-1 &&   (src(y2+1, x2-1)!=0))
						RG_PUSH2(i+w-1);
					if (x2<w-1 && y2>0 &&   (src(y2-1, x2+1)!=0))
						RG_PUSH2(i-w+1);
					if (x2<w-1 && y2<h-1 &&   (src(y2+1, x2+1)!=0))
						RG_PUSH2(i+w+1);

				}

				if (sp3 >= iMinLength)
				{
					vector<Point> component;
					component.reserve(sp3);

					// etichetto il punto
					for (i=0; i<sp3; i++){
						// etichetto
						component.push_back(stack3[i]);
					}
					segments.push_back(component);
				}
			}
		}
	}
};




void LabelingRect(Mat1b& image, VVP& segments, int iMinLength, vector<Rect>& bboxes)
{

	#define _RG_STACK_SIZE 10000

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	int stack2[_RG_STACK_SIZE];
	#define _RG_PUSH2(a) (stack2[sp2] = (a) , sp2++)
	#define _RG_POP2(a) (sp2-- , (a) = stack2[sp2])

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	Point stack3[_RG_STACK_SIZE];
	#define _RG_PUSH3(a) (stack3[sp3] = (a) , sp3++)
	#define _RG_POP3(a) (sp3-- , (a) = stack3[sp3])

	int i,w,h, iDim;
	int x,y;
	int x2,y2;	
	int sp2; /* stack pointer */
    int sp3;

	Mat_<uchar> src = image.clone();
	w = src.cols;
	h = src.rows;
	iDim = w*h;

	Point point;
	for (y=0; y<h; y++)
	{
		for (x=0; x<w; x++)
		{
			if ((src(y,x))!=0)   //punto non etichettato: seme trovato
			{
				// per ogni oggetto	
				sp2 = 0;
				i = x + y*w;
				_RG_PUSH2(i);

				// vuoto la lista dei punti
	    		sp3=0;
  		  		while (sp2>0) 
				{// rg tradizionale
		
					_RG_POP2(i);
					x2=i%w;
					y2=i/w;

					src(y2,x2) = 0;

					point.x=x2;
					point.y=y2;
					_RG_PUSH3(point);

					// Inserisco i nuovi punti nello stack solo se esistono
					// e sono punti da etichettare

					// 4 connessi
					// sx
					if (x2>0 &&   (src(y2, x2-1)!=0))
						_RG_PUSH2(i-1);
					// sotto
					if (y2>0 &&   (src(y2-1, x2)!=0))
						_RG_PUSH2(i-w);
					// sopra
					if (y2<h-1 &&   (src(y2+1, x2)!=0))
						_RG_PUSH2(i+w);
					// dx
					if (x2<w-1 &&   (src(y2, x2+1)!=0))
						_RG_PUSH2(i+1);

					// 8 connessi
					if (x2>0 && y2>0 &&   (src(y2-1,x2-1)!=0))
						_RG_PUSH2(i-w-1);
					if (x2>0 && y2<h-1 &&   (src(y2+1, x2-1)!=0))
						_RG_PUSH2(i+w-1);
					if (x2<w-1 && y2>0 &&   (src(y2-1, x2+1)!=0))
						_RG_PUSH2(i-w+1);
					if (x2<w-1 && y2<h-1 &&   (src(y2+1, x2+1)!=0))
						_RG_PUSH2(i+w+1);

				}

				if (sp3 >= iMinLength)
				{
					vector<Point> component;

					int iMinx, iMaxx, iMiny,iMaxy;
					iMinx = iMaxx = stack3[0].x;
					iMiny = iMaxy = stack3[0].y;

					// etichetto il punto
					for (i=0; i<sp3; i++){
						point = stack3[i];
						// etichetto
						component.push_back(point);

						if (iMinx > point.x)  iMinx = point.x;
						if (iMiny > point.y)  iMiny = point.y;
						if (iMaxx < point.x)  iMaxx = point.x;
						if (iMaxy < point.y)  iMaxy = point.y;
					}

					bboxes.push_back(Rect(Point(iMinx, iMiny), Point(iMaxx+1, iMaxy+1)));
					segments.push_back(component);
					
				}
			}
		}	
	}
}



// Thinning Zhang e Suen 
void Thinning(Mat1b& imgMask, uchar byF, uchar byB) 
{
	int r = imgMask.rows;
	int c = imgMask.cols;

	Mat_<uchar> imgIT(r,c),imgM(r,c);

	for(int i=0; i<r; ++i)
	{		
		for(int j=0; j<c; ++j)
		{
			imgIT(i,j) = imgMask(i,j)==byF?1:0;
		}
	}

	bool bSomethingDone = true;
	int iCount = 0;

	while (bSomethingDone) {
		bSomethingDone = false;
		fill(imgM.begin(), imgM.end(), 0);

		//prima iterazione
		for(int y=1;y<r-2;y++) {
			for(int x=1;x<c-2;x++) {

#define c_P0 imgIT(y-1,x-1)==1
#define c_P1 imgIT(y-1,x)==1
#define c_P2 imgIT(y-1,x+1)==1
#define c_P3 imgIT(y-1,x+2)==1
#define c_P4 imgIT(y,x-1)==1
#define c_P5 imgIT(y,x)==1
#define c_P6 imgIT(y,x+1)==1
#define c_P7 imgIT(y,x+2)==1
#define c_P8 imgIT(y+1,x-1)==1
#define c_P9 imgIT(y+1,x)==1
#define c_P10 imgIT(y+1,x+1)==1
#define c_P11 imgIT(y+1,x+2)==1
#define c_P12 imgIT(y+2,x-1)==1
#define c_P13 imgIT(y+2,x)==1
#define c_P14 imgIT(y+2,x+1)==1
#define c_P15 imgIT(y+2,x+2)==1

				if (c_P5) {
					if (c_P9) {
						if (c_P6) {
							if (c_P10) {
								if (c_P4) {
									if (c_P8) {
										if (c_P1) {
											continue;
										}
										else {
											if (c_P13) {
												if (c_P2) {
													if (c_P0) {
														continue;
													}
													else {
														goto a_2;
													}
												}
												else {
													goto a_2;
												}
											}
											else {
												if (c_P14) {
													if (c_P12) {
														if (c_P2) {
															if (c_P0) {
																continue;
															}
															else {
																goto a_2;
															}
														}
														else {
															goto a_2;
														}
													}
													else {
														continue;
													}
												}
												else {
													continue;
												}
											}
										}
									}
									else {
										continue;
									}
								}
								else {
									if (c_P1) {
										if (c_P2) {
											if (c_P7) {
												if (c_P8) {
													if (c_P0) {
														continue;
													}
													else {
														goto a_2;
													}
												}
												else {
													goto a_2;
												}
											}
											else {
												if (c_P11) {
													if (c_P3) {
														if (c_P8) {
															if (c_P0) {
																continue;
															}
															else {
																goto a_2;
															}
														}
														else {
															goto a_2;
														}
													}
													else {
														continue;
													}
												}
												else {
													continue;
												}
											}
										}
										else {
											continue;
										}
									}
									else {
										if (c_P0) {
											continue;
										}
										else {
											if (c_P8) {
												goto a_2;
											}
											else {
												if (c_P2) {
													goto a_2;
												}
												else {
													if (c_P14) {
														if (c_P13) {
															if (c_P11) {
																goto a_2;
															}
															else {
																if (c_P7) {
																	goto a_2;
																}
																else {
																	if (c_P3) {
																		goto a_2;
																	}
																	else {
																		continue;
																	}
																}
															}
														}
														else {
															goto a_2;
														}
													}
													else {
														if (c_P13) {
															goto a_2;
														}
														else {
															if (c_P12) {
																goto a_2;
															}
															else {
																if (c_P11) {
																	if (c_P7) {
																		continue;
																	}
																	else {
																		goto a_2;
																	}
																}
																else {
																	if (c_P15) {
																		goto a_2;
																	}
																	else {
																		if (c_P7) {
																			goto a_2;
																		}
																		else {
																			if (c_P3) {
																				goto a_2;
																			}
																			else {
																				continue;
																			}
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
							else {
								continue;
							}
						}
						else {
							if (c_P0) {
								if (c_P8) {
									if (c_P4) {
										if (c_P2) {
											if (c_P10) {
												continue;
											}
											else {
												if (c_P1) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
										else {
											goto a_2;
										}
									}
									else {
										continue;
									}
								}
								else {
									continue;
								}
							}
							else {
								if (c_P2) {
									continue;
								}
								else {
									if (c_P1) {
										continue;
									}
									else {
										if (c_P8) {
											goto a_2;
										}
										else {
											if (c_P10) {
												if (c_P4) {
													continue;
												}
												else {
													goto a_2;
												}
											}
											else {
												continue;
											}
										}
									}
								}
							}
						}
					}
					else {
						if (c_P6) {
							if (c_P0) {
								if (c_P2) {
									if (c_P1) {
										if (c_P8) {
											if (c_P10) {
												continue;
											}
											else {
												if (c_P4) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
										else {
											goto a_2;
										}
									}
									else {
										continue;
									}
								}
								else {
									continue;
								}
							}
							else {
								if (c_P8) {
									continue;
								}
								else {
									if (c_P4) {
										continue;
									}
									else {
										if (c_P2) {
											goto a_2;
										}
										else {
											if (c_P10) {
												if (c_P1) {
													continue;
												}
												else {
													goto a_2;
												}
											}
											else {
												continue;
											}
										}
									}
								}
							}
						}
						else {
							if (c_P10) {
								continue;
							}
							else {
								if (c_P4) {
									if (c_P1) {
										if (c_P0) {
											goto a_2;
										}
										else {
											continue;
										}
									}
									else {
										if (c_P2) {
											continue;
										}
										else {
											if (c_P8) {
												goto a_2;
											}
											else {
												if (c_P0) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
									}
								}
								else {
									if (c_P8) {
										continue;
									}
									else {
										if (c_P1) {
											if (c_P2) {
												goto a_2;
											}
											else {
												if (c_P0) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
										else {
											continue;
										}
									}
								}
							}
						}
					}
				}
				else {
					continue;
				}


a_2:
				imgM(y,x) = 1;
				bSomethingDone = true;
			}
		}
		
		for (int r=0; r<imgIT.rows; ++r) {
			for (int c=0; c<imgIT.cols; ++c) {
				if (imgM(r,c) == 1)
					imgIT(r,c) = 0;
			}
		}
	}

	for(int i=0; i<r; ++i)
	{		
		for(int j=0; j<c; ++j)
		{
			imgMask(i,j) = imgIT(i,j)==1 ? byF : byB;
		}
	}
};

bool SortBottomLeft2TopRight(const Point& lhs, const Point& rhs)
{
	if(lhs.x == rhs.x)
	{
		return lhs.y > rhs.y;
	}
	return lhs.x < rhs.x;
};

bool SortBottomLeft2TopRight2f(const Point2f& lhs, const Point2f& rhs)
{
	if(lhs.x == rhs.x)
	{
		return lhs.y > rhs.y;
	}
	return lhs.x < rhs.x;
};


bool SortTopLeft2BottomRight(const Point& lhs, const Point& rhs)
{
	if(lhs.x == rhs.x)
	{
		return lhs.y < rhs.y;
	}
	return lhs.x < rhs.x;
};




void Canny3(InputArray image, OutputArray _edges,
			OutputArray _sobel_x, OutputArray _sobel_y,
                int apertureSize, bool L2gradient )
{
    Mat src = image.getMat();
    _edges.create(src.size(), CV_8U);
	_sobel_x.create(src.size(), CV_16S);
	_sobel_y.create(src.size(), CV_16S);

    // Use OpenCV 4's Canny implementation with automatic thresholding
    Sobel(src, _sobel_x, CV_16S, 1, 0, apertureSize);
    Sobel(src, _sobel_y, CV_16S, 0, 1, apertureSize);
    
    // Calculate gradient magnitude - need to convert to float first
    Mat sobel_x_float, sobel_y_float;
    _sobel_x.getMat().convertTo(sobel_x_float, CV_32F);
    _sobel_y.getMat().convertTo(sobel_y_float, CV_32F);
    
    Mat magGrad;
    magnitude(sobel_x_float, sobel_y_float, magGrad);
    
    // Normalize and convert to 8-bit for Otsu's thresholding
    Mat magGrad_8u;
    normalize(magGrad, magGrad_8u, 0, 255, NORM_MINMAX, CV_8U);
    
    // Automatic thresholding using Otsu's method
    double high_thresh = threshold(magGrad_8u, magGrad_8u, 0, 255, THRESH_BINARY | THRESH_OTSU);
    double low_thresh = high_thresh * 0.3;
    
    Canny(_sobel_x, _sobel_y, _edges, low_thresh, high_thresh, L2gradient);
};




float GetMinAnglePI(float alpha, float beta)
{
	float pi = float(CV_PI);
	float pi2 = float(2.0 * CV_PI);

	//normalize data in [0, 2*pi]
	float a = fmod(alpha + pi2, pi2);
	float b = fmod(beta + pi2, pi2);

	//normalize data in [0, pi]
	if (a > pi)
		a -= pi;
	if (b > pi)
		b -= pi;

	if (a > b)
	{
		swap(a, b);
	}

	float diff1 = b - a;
	float diff2 = pi - diff1;
	return min(diff1, diff2);
}