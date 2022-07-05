// Gmsh project created on Wed Dec 22 20:54:51 2021
SetFactory("OpenCASCADE");
//
Rinf = DefineNumber[ 0.5, Name "Parameters/Rinf" ];
Rs = DefineNumber[ 0.02, Name "Parameters/Rs" ];
//
meshmult = 0.5;		// mesh refinement multiplication parameter (bigger = finer)

// Define 2 circles, one on the infinity boundary and other surrounding the surrogate area
// This is done to be able to define different mesh sizes on those lines

Circle(1) = {0, 0, 0, Rinf};
Circle(2) = {0, 0, 0, Rs};
// Using Transfinite definition to set the number of points in the circles
//Transfinite Curve {1} = meshmult*50 Using Progression 1;
//Transfinite Curve {2} = meshmult*50 Using Progression 1;

// Create Curve loops from circles to be able to use them to get plane surfaces

Curve Loop (3) = {1};
Curve Loop (4) = {2};

//-------------------------------------------------------------------------
// Pattern creation or square surrogate substitution
// Patter type: 9 cilinders in a 3x3 matrix
//
// angle to rotate geometry to simulate velocity angle variation
// line can be automatically edited by scripts
vAngle = 0;
//
lpatt = 0.01;           // ortogonal distance between elemnts in the matrix
rpatt = 0.002;          // radius of obstacle cylinders
//--- Variation for the pattern of obstacles start here
Xini = -lpatt;          // leftmost lower coordinate in the pattern
Yini = -lpatt;
CirPattList[] ={};      // List of Circle Line to export as boundaries
CurLooPattList[] = {};  // List of Curve Loops for obstacles
For i In {1:9}
    Xp = Xini + Modulo( (i - 1) , 3 ) * lpatt;  // location of the obstacles
    Yp = Yini + Floor( (i - 1) / 3 ) * lpatt;
    Xc = Xp * Cos(-vAngle) - Yp * Sin(-vAngle);
    Yc = Xp * Sin(-vAngle) + Yp * Cos(-vAngle);
    Circle (4+i) = {Xc, Yc, 0, rpatt};
    CirPattList[] += 4+i;
    Curve Loop (13+i) = {4+i};
    CurLooPattList[] += 13+i;
EndFor
//--- End of Variation
//--- Variation for the surrogate model
// Square enclosing area for the 3x3 matrix pattern
// From (- lpatt - rpatt, - lpatt - rpatt) to (lpatt + rpatt, lpatt + rpatt)
// VerXY = lpatt + rpatt;   // Square vertices are symmetrical from origin
// Point(1) = {-VerXY, -VerXY, 0};
// Point(2) = {-VerXY, VerXY,  0};
// Point(3) = {VerXY,  VerXY,  0};
// Point(4) = {VerXY,  -VerXY, 0};
// Line(3) = {1, 2};
// Line(4) = {2, 3};
// Line(5) = {3, 4};
// Line(6) = {4, 1};
// Curve Loop (5) = {3, 4, 5, 6};
// CurLooPattList[] = {5};
//--- End of Variation
//-------------------------------------------------------------------------

// Create the plane surfaces (Same lines can be used for both cases

Plane Surface(1) = {3, 4};
Plane Surface(2) = {4, CurLooPattList[]};
// Uncomment when using surrogate substitution
// Plane Surface(3) = {CurLooPattList[]};

// Using mesh size applied to the points of the circles
MeshSize{ PointsOf{ Curve{1}; } }                   = 0.02  / meshmult;
MeshSize{ PointsOf{ Curve{2}; } }                   = 0.001 / meshmult;
// Use this line in case of using the matrix pattern...
MeshSize{ PointsOf{ Curve{CirPattList[]}; } }       = 0.0002/ meshmult;
// ... or this in case of using the surrogate square
// MeshSize{ PointsOf{ Curve{5}; } }                   = 0.001 / meshmult;

//Recombine Surface {1};
//Recombine Surface {2};

// Physical entities definition
Physical Line(     "Infty_B")   	= {1};
//Physical Line(     "Surr_B")    	= {2};
Physical Line(     "Obst_B")    	= {CirPattList[]};
Physical Surface(   "OutDomain")        = {1};
Physical Surface(   "InDomain")         = {2};
// Uncomment when using square surrogate
// Physical Surface(   "Surrogate")        = {3};
