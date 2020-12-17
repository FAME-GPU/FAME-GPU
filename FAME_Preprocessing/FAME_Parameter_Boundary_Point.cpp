#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Parameter_Boundary_Point(LATTICE* Lattice, MESH Mesh)
{
    int i;
    double R3R8 = 0.0, R10R3 = 0.0;// R9R1 = 0.0, R11R2 = 0.0;
    // initialize t1, t2, t3, t4 as zero vectors
    Lattice->lattice_constant.t1 = (double*) calloc(3, sizeof(double));
    Lattice->lattice_constant.t2 = (double*) calloc(3, sizeof(double));
    Lattice->lattice_constant.t3 = (double*) calloc(3, sizeof(double));
    Lattice->lattice_constant.t4 = (double*) calloc(3, sizeof(double));

    if( (Lattice->lattice_constant.theta_3 > 0)  &&  (Lattice->lattice_constant.theta_3 <= pi/2) )
        R3R8 = Lattice->lattice_vec_a[3];
    else
        R3R8 = Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[3];

    // Case 1(i-iv)
    if( (Lattice->lattice_vec_a[6] > 0)  &&  (Lattice->lattice_vec_a[7] > 0) ) // if a3(1) > 0  &&  a3(2) > 0
    {
        //R9R1 = Lattice->lattice_vec_a[6];  // R9R1  = a3(1);
        R10R3 = Lattice->lattice_vec_a[7]; // R10R3 = a3(2);

        for(i = 0; i < 3; i++)
        {
            Lattice->lattice_constant.t1[i] = 0; // t1 = [0 0 0];
            Lattice->lattice_constant.t2[i] = -Lattice->lattice_vec_a[i]; // t2 = -a1;
        }
        // case(1-i): if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) <= a3(1)
        if ( (Lattice->lattice_constant.theta_3 > 0)     && \
             (Lattice->lattice_constant.theta_3 <= pi/2)  && \
             (Lattice->lattice_vec_a[3] <= Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "1-i");
            //R11R2 = Lattice->lattice_vec_a[6] - Lattice->lattice_vec_a[3]; // R11R2 = a3(1) - a2(1);

            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = -Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t3 = -a1 - a2;
                Lattice->lattice_constant.t4[i] = -Lattice->lattice_vec_a[i + 3]; // t4 = -a2;
            }
        }
        // case(1-ii): if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) > a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > 0)     && \
                  (Lattice->lattice_constant.theta_3 <= pi/2)  && \
                  (Lattice->lattice_vec_a[3] > Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "1-ii");
            //R11R2 = Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R11R2 = a1(1) -  a2(1) + a3(1);

            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = -Lattice->lattice_vec_a[i + 3]; // t3 = -a2;
                Lattice->lattice_constant.t4[i] = Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t4 = a1 - a2;
            }
        }
        // case(1-iii): if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) <= a1(1)-a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2)     && \
                  (Lattice->lattice_constant.theta_3 < pi  )     && \
                  (-1*Lattice->lattice_vec_a[3] <= Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "1-iii");
            //R11R2 = Lattice->lattice_vec_a[6] - Lattice->lattice_vec_a[3]; // R11R2 = a3(1) - a2(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = -Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t3 = -a1 - a2;
                Lattice->lattice_constant.t4[i] = -Lattice->lattice_vec_a[i + 3]; // t4 = -a2;
            }
        }
        // case(1-iv): if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) >  a1(1)-a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2)     && \
                  (Lattice->lattice_constant.theta_3 < pi  )     && \
                  (-1*Lattice->lattice_vec_a[3] > Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "1-iv");
            //R11R2 = -Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[6] - Lattice->lattice_vec_a[3]; // R11R2 = -a1(1) + a3(1) - a2(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = -2 * Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t3 = -2a1 - a2;
                Lattice->lattice_constant.t4[i] = -Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t4 = -a1 - a2;
            }
        }
    }

    // Case 2(i-iv)
    if( (Lattice->lattice_vec_a[6] < 0)  &&  (Lattice->lattice_vec_a[7] > 0) ) // if a3(1) < 0  &&  a3(2) > 0
    {
        //R9R1 = Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[6]; // R9R1 = a1(1) + a3(1);
        R10R3 = Lattice->lattice_vec_a[7]; // R10R3 = a3(2);

        for(i = 0; i < 3; i++)
        {
            Lattice->lattice_constant.t1[i] = Lattice->lattice_vec_a[i]; // t1 = a1;
            Lattice->lattice_constant.t2[i] = 0; // t2 = [0 0 0];
        }
        // case(2-i) if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) <= a1(1)+a3(1)
        if ( (Lattice->lattice_constant.theta_3 > 0     )     && \
             (Lattice->lattice_constant.theta_3 <= pi/2  )     && \
             (Lattice->lattice_vec_a[3] <= Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "2-i");
            //R11R2 = Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R11R2 = a1(1) - a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = -Lattice->lattice_vec_a[i + 3]; // t3 = -a2;
                Lattice->lattice_constant.t4[i] = Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t4 = a1 - a2;
            }
        }
        // case(2-ii) if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) >  a1(1)+a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > 0     )     && \
                  (Lattice->lattice_constant.theta_3 <= pi/2  )     && \
                  (Lattice->lattice_vec_a[3] > Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "2-ii");
            //R11R2 = 2 * Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R11R2 = 2a1(1) - a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t3 = a1 - a2;
                Lattice->lattice_constant.t4[i] = 2 * Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t4 = 2a1 - a2;
            }
        }
        // case(2-iii) if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) <= -a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2  )     && \
                  (Lattice->lattice_constant.theta_3 < pi    )     && \
                  (-1*Lattice->lattice_vec_a[3] <= -1*Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "2-iii");
            //R11R2 = Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R11R2 = a1(1) - a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = -Lattice->lattice_vec_a[i + 3]; // t3 = -a2;
                Lattice->lattice_constant.t4[i] = Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t4 = a1 - a2;
            }
        }
        // case(2-iv) if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) >  -a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2  )     && \
                  (Lattice->lattice_constant.theta_3 < pi    )     && \
                  (-1*Lattice->lattice_vec_a[3] > -1*Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "2-iv");
            //R11R2 = -Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R11R2 = -a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t3[i] = -Lattice->lattice_vec_a[i] - Lattice->lattice_vec_a[i + 3]; // t3 = -a1 - a2;
                Lattice->lattice_constant.t4[i] = - Lattice->lattice_vec_a[i + 3]; // t4 = -a2;
            }
        }
    }

    // Case 3(i-iv)
    if( (Lattice->lattice_vec_a[6] < 0)  &&  (Lattice->lattice_vec_a[7] < 0) ) // if a3(1) < 0  &&  a3(2) < 0
    {
        R10R3 = Lattice->lattice_vec_a[4] + Lattice->lattice_vec_a[7]; // R10R3 = a2(2) + a3(2);
        //R11R2 = Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[6]; // R11R2 = a1(1) + a3(1);

        for(i = 0; i < 3; i++)
        {
            Lattice->lattice_constant.t3[i] = 0; // t3 = [0 0 0];
            Lattice->lattice_constant.t4[i] = Lattice->lattice_vec_a[i]; // t4 = a1;
        }
        // case(3-i) if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) <= -a3(1)
        if ( (Lattice->lattice_constant.theta_3 > 0       )     && \
             (Lattice->lattice_constant.theta_3 <= pi/2    )     && \
             (Lattice->lattice_vec_a[3] <= -1*Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "3-i");
            //R9R1 = -Lattice->lattice_vec_a[3] - Lattice->lattice_vec_a[6]; // R9R1 = -a2(1) - a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t1 = a1 + a2;
                Lattice->lattice_constant.t2[i] = Lattice->lattice_vec_a[i + 3]; // t2 = a2;
            }
        }
        // case(3-ii) if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) > -a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > 0       )     && \
             (Lattice->lattice_constant.theta_3 <= pi/2    )     && \
             (Lattice->lattice_vec_a[3] > -1*Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "3-ii");
            //R9R1 = Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R9R1 = a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = Lattice->lattice_vec_a[i + 3] + Lattice->lattice_vec_a[i + 3]; // t1 = a2;
                Lattice->lattice_constant.t2[i] = - Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t2 = -a1 + a2;
            }
        }
        // case(3-iii) if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) <= a1(1)+a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2  )     && \
             (Lattice->lattice_constant.theta_3 < pi    )     && \
             (-1*Lattice->lattice_vec_a[3] <= Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "3-iii");
            //R9R1 = Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R9R1 = a1(1) + a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t1 = a1 + a2;
                Lattice->lattice_constant.t2[i] = Lattice->lattice_vec_a[i + 3]; // t2 = a2;
            }
        }
        // case(3-iv) if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) > a1(1)+a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2  )     && \
             (Lattice->lattice_constant.theta_3 < pi    )     && \
             (-1*Lattice->lattice_vec_a[3] > Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "3-iv");
            //R9R1 = 2 * Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R9R1 = 2a1(1) + a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = 2 * Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t1 = 2a1 + a2;
                Lattice->lattice_constant.t2[i] = Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t2 = a1 + a2;
            }
        }
    }

    // Case 4(i-iv)
    if( (Lattice->lattice_vec_a[6] > 0)  &&  (Lattice->lattice_vec_a[7] < 0) ) // a3(1) > 0  &&  a3(2) < 0
    {
        R10R3 = Lattice->lattice_vec_a[4] + Lattice->lattice_vec_a[7]; // R10R3 = a2(2) + a3(2);
        //R11R2 = Lattice->lattice_vec_a[6]; // R11R2 = a3(1);
        for(i = 0; i < 3; i++)
        {
            Lattice->lattice_constant.t3[i] = -Lattice->lattice_vec_a[i]; // t3 = -a1;
            Lattice->lattice_constant.t4[i] = 0; // t4 = [0 0 0];
        }
        // case(4-i) if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) <= a1(1)-a3(1)
        if ( (Lattice->lattice_constant.theta_3 > 0       )     && \
             (Lattice->lattice_constant.theta_3 <= pi/2    )     && \
             (Lattice->lattice_vec_a[3] <= Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "4-i");
            //R9R1 = Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R9R1 = a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = Lattice->lattice_vec_a[i + 3]; // t1 = a2;
                Lattice->lattice_constant.t2[i] = -Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t2 = -a1 + a2;
            }
        }
        // case(4-ii) if (lattice_constant.theta_3 > 0  &&  lattice_constant.theta_3 < pi/2) && a2(1) >  a1(1)-a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > 0       )     && \
                  (Lattice->lattice_constant.theta_3 <= pi/2    )     && \
                  (Lattice->lattice_vec_a[3] > Lattice->lattice_vec_a[0] - Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "4-ii");
            //R9R1 = -Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R9R1 = -a1(1) + a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = -Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t1 = -a1 + a2;
                Lattice->lattice_constant.t2[i] = -2 * Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t2 = -2a1 + a2;
            }
        }
        // case(4-iii) if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) <= a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2  )     && \
                  (Lattice->lattice_constant.theta_3 < pi    )     && \
                  (-1*Lattice->lattice_vec_a[3] <= Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "4-iii");
            //R9R1 = Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R9R1 = a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = Lattice->lattice_vec_a[i + 3]; // t1 = a2;
                Lattice->lattice_constant.t2[i] = -Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t2 = -a1 + a2;
            }
        }
        // case(4-iv) if (lattice_constant.theta_3 > pi/2  &&  lattice_constant.theta_3 < pi) && -a2(1) > a3(1)
        else if ( (Lattice->lattice_constant.theta_3 > pi/2  )     && \
                  (Lattice->lattice_constant.theta_3 < pi    )     && \
                  (-1*Lattice->lattice_vec_a[3] > Lattice->lattice_vec_a[6] ) )
        {
            strcpy(Lattice->lattice_constant.flag, "4-iv");
            //R9R1 = Lattice->lattice_vec_a[0] + Lattice->lattice_vec_a[3] + Lattice->lattice_vec_a[6]; // R9R1 = a1(1) + a2(1) + a3(1);
            for(i = 0; i < 3; i++)
            {
                Lattice->lattice_constant.t1[i] = Lattice->lattice_vec_a[i] + Lattice->lattice_vec_a[i + 3]; // t1 = a1 + a2;
                Lattice->lattice_constant.t2[i] = Lattice->lattice_vec_a[i + 3]; // t2 = a2;
            }
        }
    }

    Lattice->lattice_constant.m1 = round( R3R8/Mesh.mesh_len[0] );
    Lattice->lattice_constant.m2 = round( (Lattice->lattice_vec_a[6] + Lattice->lattice_constant.t1[0]) / Mesh.mesh_len[0] ) ;
    Lattice->lattice_constant.m3 = round( R10R3/Mesh.mesh_len[1]) ;
    Lattice->lattice_constant.m4 = (Lattice->lattice_constant.m2 - Lattice->lattice_constant.m1) % Mesh.grid_nums[0]; // mod(m2 - m1, grid_num(1)) ;
    if ( Lattice->lattice_constant.m4 < 0 )
        Lattice->lattice_constant.m4 = Lattice->lattice_constant.m4 + Mesh.grid_nums[0];
    
    return 0;
}