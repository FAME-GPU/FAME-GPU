#include "FAME_Internal_Common.h"
#include "FAME_Parameter_Brillouin_Zone_Point.h"
// 2020-02-19

void confirm_wave_vec_array_size(char* path_string,int* wave_vec_num, int part_num);
void default_path(LATTICE lattice, RECIP_LATTICE* recip_lattice);
void get_vertex_pt(RECIP_LATTICE *recip_lattice, double* subpath_start_string, double* subpath_end_string, int i);

int FAME_Parameter_Brillouin_Zone_Path(RECIP_LATTICE* Recip_lattice, int part_num, LATTICE Lattice)
{
    int str_len, count_part, flag, count;
    double part_x, part_y, part_z;
    double subpath_start_string[3];
    double subpath_end_string[3];
    double subpath[3][part_num];
    default_path(Lattice, Recip_lattice);
    confirm_wave_vec_array_size(Recip_lattice->path_string, &Recip_lattice->Wave_vec_num, part_num);

    Recip_lattice->WaveVector = (double*) calloc(3 * Recip_lattice->Wave_vec_num, sizeof(double));

    //printf("= = = = = = FAME_Parameter_Brillouin_Zone_Point = = = = = = = = = = = = = = = = =\n");
    FAME_Parameter_Brillouin_Zone_Point( &(Recip_lattice->vertex), Lattice, Recip_lattice->reciprocal_lattice_vector_b);

    str_len    = strlen(Recip_lattice->path_string);
    count_part = 0;
    flag       = 0;
    count      = 0;

    for ( int i=0; i < str_len-1; i++ )
    {
        if ( (Recip_lattice->path_string[i] != '|') && (Recip_lattice->path_string[i+1] !='|') )
        {
            get_vertex_pt( Recip_lattice, subpath_start_string, subpath_end_string, i);
            part_x = (subpath_end_string[0] - subpath_start_string[0])/(double)(part_num-1);
            part_y = (subpath_end_string[1] - subpath_start_string[1])/(double)(part_num-1);
            part_z = (subpath_end_string[2] - subpath_start_string[2])/(double)(part_num-1);
            for (int j = 0; j < part_num; j++ )
            {
                subpath[0][j] = subpath_start_string[0] + part_x*j;
                subpath[1][j] = subpath_start_string[1] + part_y*j;
                subpath[2][j] = subpath_start_string[2] + part_z*j;
            }
            if ( i == (str_len-2) || (Recip_lattice->path_string[i+2] == '|') )
            {
                for (int j = 0; j < part_num; j++ )
                {
                    Recip_lattice->WaveVector[count * 3 + 0] = subpath[0][j];
                    Recip_lattice->WaveVector[count * 3 + 1] = subpath[1][j];
                    Recip_lattice->WaveVector[count * 3 + 2] = subpath[2][j];
                    count = count+1;
                }
            }
            else
            {
                for (int j = 0; j < part_num-1; j++ )
                {
                    Recip_lattice->WaveVector[count * 3 + 0] = subpath[0][j];
                    Recip_lattice->WaveVector[count * 3 + 1] = subpath[1][j];
                    Recip_lattice->WaveVector[count * 3 + 2] = subpath[2][j];
                    count = count+1;
                }
            }

        }
        if ( Recip_lattice->path_string[i] == '|')
        {
            count_part = count_part+1;
            flag  = 1;
        }
        else if (Recip_lattice->path_string[i+1] == '|')
        {
            continue;
        }
        else if (flag == 1)
        {
            flag = 0;
        }
    }

    double V1, V2, V3;
    for(int i = 0; i < Recip_lattice->Wave_vec_num; i++)
    {
        V1 = Recip_lattice->WaveVector[i * 3 + 0];
        V2 = Recip_lattice->WaveVector[i * 3 + 1];
        V3 = Recip_lattice->WaveVector[i * 3 + 2];

        Recip_lattice->WaveVector[i * 3 + 0] = Lattice.Omega[0] * V1 + Lattice.Omega[3] * V2 + Lattice.Omega[6] * V3;
        Recip_lattice->WaveVector[i * 3 + 1] = Lattice.Omega[1] * V1 + Lattice.Omega[4] * V2 + Lattice.Omega[7] * V3;
        Recip_lattice->WaveVector[i * 3 + 2] = Lattice.Omega[2] * V1 + Lattice.Omega[5] * V2 + Lattice.Omega[8] * V3;
    }

    return 0;
}
/*
int FAME_Parameter_Brillouin_Zone_Path(RECIP_LATTICE* Recip_lattice, int part_num, LATTICE Lattice, char* path_string )
{
    int n, count_part, flag, count;
    double part_x, part_y, part_z;
    double subpath_start_string[3];
    double subpath_end_string[3];
    double subpath[3][part_num];

    confirm_wave_vec_array_size(path_string, &Recip_lattice->Wave_vec_num, part_num);
    Recip_lattice->WaveVector = (double*) calloc(3 * Recip_lattice->Wave_vec_num, sizeof(double));

    FAME_Parameter_Brillouin_Zone_Point( &(Recip_lattice->vertex), Lattice, Recip_lattice->reciprocal_lattice_vector_b);

    n = strlen(path_string);
    count_part = 0;
    flag = 0;
    count = 0;

    for (int i=0; i<n-1; i++)
    {
        if ( (path_string[i] != '|') && (path_string[i+1] !='|') )
        {
            get_vertex_pt( Recip_lattice, subpath_start_string, subpath_end_string, i);
            part_x = (subpath_end_string[0]-subpath_start_string[0])/(double)(part_num-1);
            part_y = (subpath_end_string[1]-subpath_start_string[1])/(double)(part_num-1);
            part_z = (subpath_end_string[2]-subpath_start_string[2])/(double)(part_num-1);
            for (int j=0; j<part_num; j++)
            {
                subpath[0][j] = subpath_start_string[0]+part_x*j;
                subpath[1][j] = subpath_start_string[1]+part_y*j;
                subpath[2][j] = subpath_start_string[2]+part_z*j;
            }
            if ( i==(n-2) || (path_string[i+2] == '|') )
            {
                for (int j=0; j<part_num; j++)
                {
                    Recip_lattice->WaveVector[count * 3 + 0] = subpath[0][j];
                    Recip_lattice->WaveVector[count * 3 + 1] = subpath[1][j];
                    Recip_lattice->WaveVector[count * 3 + 2] = subpath[2][j];
                    count = count+1;
                }
            }
            else
            {
                for (int j=0; j<part_num-1; j++)
                {
                    Recip_lattice->WaveVector[count * 3 + 0] = subpath[0][j];
                    Recip_lattice->WaveVector[count * 3 + 1] = subpath[1][j];
                    Recip_lattice->WaveVector[count * 3 + 2] = subpath[2][j];
                    count = count+1;
                }
            }
        }
        if ( path_string[i] == '|')
        {
            count_part = count_part+1;
            flag  = 1;
        }
        else if (path_string[i+1] == '|')
        {
            continue;
        }
        else if (flag == 1)
        {
            flag = 0;
        }
    }

    return 0;
}
*/
void default_path(LATTICE lattice, RECIP_LATTICE* recip_lattice)
{
    if(strcmp(lattice.lattice_type, "simple_cubic") == 0)
    {
        strcpy(recip_lattice->path_string, "GXMGRX|MR");
    }
    else if(strcmp(lattice.lattice_type, "face_centered_cubic") == 0)
    {
        strcpy(recip_lattice->path_string, "GXWKGLUWLK|UX");
    }
    else if(strcmp(lattice.lattice_type, "body_centered_cubic") == 0)
    {
        strcpy(recip_lattice->path_string, "GHNGPH|PN");
    }
    else if(strcmp(lattice.lattice_type, "hexagonal") == 0)
    {
        strcpy(recip_lattice->path_string, "GMKGALHA|LM|KH");
    }
    else if(strcmp(lattice.lattice_type, "rhombohedral") == 0)
    {
        double theta = 2*asin( 3*(lattice.lattice_constant.a)/(2*sqrt( pow(lattice.lattice_constant.c, 2) + 3*(pow(lattice.lattice_constant.a, 2)))));
        if(theta <= pi/2)
            strcpy(recip_lattice->path_string, "GLC|BZGX|QFRZ|LP");
        else
            strcpy(recip_lattice->path_string, "GPZQGFRSLZ");
    }
    else if(strcmp(lattice.lattice_type, "primitive_tetragonal") == 0)
    {
        strcpy(recip_lattice->path_string, "GXMGZRAZ|XR|MA");
    }
    else if(strcmp(lattice.lattice_type, "body_centered_tetragonal") == 0)
    {
        if((lattice.lattice_constant.c)<(lattice.lattice_constant.a))
            strcpy(recip_lattice->path_string, "GXMGZPNBM|XP");
        else if((lattice.lattice_constant.c)>(lattice.lattice_constant.a))
            strcpy(recip_lattice->path_string, "GXYsGZtNPAZ|XP");
    }
    else if(strcmp(lattice.lattice_type, "primitive_orthorhombic") == 0)
    {
        strcpy(recip_lattice->path_string, "GXSYGZURTZ|YT|UX|SR");
    }
    else if(strcmp(lattice.lattice_type, "c_base_centered_orthorhombic") == 0)
    {
        strcpy(recip_lattice->path_string, "GXSRAZGYCBTY|ZT");
    }
    else if(strcmp(lattice.lattice_type, "face_centered_orthorhombic") == 0)
    {
        double temp1 = 1/pow(lattice.lattice_constant.a, 2);
        double temp2 = 1/pow(lattice.lattice_constant.b, 2) + 1/pow(lattice.lattice_constant.c, 2);
        if(temp1>temp2)
            strcpy(recip_lattice->path_string, "GYTZGXBY|TJ|XAZ|LG");
        else if(temp1<temp2)
            strcpy(recip_lattice->path_string, "GYCDXGZFHC|EZ|XI|HY|LG");
        else
            strcpy(recip_lattice->path_string, "GYTZGXBY|XAZ|LG");
    }
    else if(strcmp(lattice.lattice_type, "body_centered_orthorhombic") == 0)
        strcpy(recip_lattice->path_string, "GXLTWRAZGYSW|IY|BZ");
    else if(strcmp(lattice.lattice_type, "primitive_monoclinic") == 0)
        strcpy(recip_lattice->path_string, "GYHCENAXI|MDZ|YD");
    else if(strcmp(lattice.lattice_type, "base_centered_monoclinic") == 0)
    {
        double temp = (lattice.lattice_constant.b)*cosf(lattice.lattice_constant.alpha)/(lattice.lattice_constant.c)+pow(lattice.lattice_constant.b, 2)*pow(sinf(lattice.lattice_constant.alpha), 2)/pow(lattice.lattice_constant.a, 2);
        int i;
        double dot = 0;
        double normb1 = 0;
        double normb2 = 0;
        double k_gamma;

        for(i=0; i<3; i++)
        {
            dot += (recip_lattice->reciprocal_lattice_vector_b)[i]*(recip_lattice->reciprocal_lattice_vector_b)[3+i];
            normb1 += pow((recip_lattice->reciprocal_lattice_vector_b)[i], 2);
            normb2 += pow((recip_lattice->reciprocal_lattice_vector_b)[i+3], 2);
        }

        normb1 = sqrt(normb1);
        normb2 = sqrt(normb2);
        k_gamma = acos(dot/(normb1*normb2));

        if(k_gamma > pi/2)
            strcpy((recip_lattice->path_string), "GYFLI|RZJ|YT|XGN|MG");
        else if(k_gamma == pi/2)
            strcpy((recip_lattice->path_string), "GYFLI|RZJ|NGM");
        else if(k_gamma < pi/2 && temp < 1)
            strcpy((recip_lattice->path_string), "GYFHZIJ|PVXGN|MG");
        else if(k_gamma < pi/2 && temp == 1)
            strcpy((recip_lattice->path_string), "GYFHZI|PVXGN|MG");
        else if(k_gamma < pi/2 && temp > 1)
            strcpy((recip_lattice->path_string), "GYFLI|RZHJ|PVXGN|MG");
    }
    else if(strcmp(lattice.lattice_type, "triclinic") == 0)
        strcpy(recip_lattice->path_string, "XGY|LGZ|NGM|RG");
    else
    {
        /*
        printf("\033[40;31mFAME_Parameter_Brillouin_Zone_Path(262):\033[0m\n");
        printf("\033[40;31mLattice type '%s' is invalid! Please check lattice type is correct in material data.\033[0m\n", lattice.lattice_type);
        assert(0);
        */
    }
}

void confirm_wave_vec_array_size(char* path_string, int* wave_vec_num, int part_num)
{
    int i,size_n,count=0,temp;
    *wave_vec_num = 0;
    size_n = strlen(path_string);
    for(i=0; i<size_n; i++)
    {
        if(path_string[i]=='|')
        {
            temp = (i-count-1)*(part_num-1)+1;
            *wave_vec_num = *wave_vec_num + temp;
            count = i + 1;
        }
        else if(i==size_n-1)
        {
            temp = (i-count)*(part_num-1)+1;
            *wave_vec_num = *wave_vec_num + temp;
        }
    }
}

void get_vertex_pt(RECIP_LATTICE *recip_lattice, double* subpath_start_string, double* subpath_end_string, int index)
{
    string path_string(recip_lattice->path_string);

    char s_p = path_string[index];
    char e_p = path_string[index + 1];

    int i;
    switch(s_p)
    {
    case 'A':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.A[i];
        break;
    case 'B':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.B[i];
        break;
    case 'C':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.C[i];
        break;
    case 'D':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.D[i];
        break;
    case 'E':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.E[i];
        break;
    case 'F':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.F[i];
        break;
    case 'G':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.G[i];
        break;
    case 'H':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.H[i];
        break;
    case 'I':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.I[i];
        break;
    case 'J':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.J[i];
        break;
    case 'K':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.K[i];
        break;
    case 'L':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.L[i];
        break;
    case 'M':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.M[i];
        break;
    case 'N':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.N[i];
        break;
    case 'O':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.O[i];
        break;
    case 'P':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.P[i];
        break;
    case 'Q':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.Q[i];
        break;
    case 'R':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.R[i];
        break;
    case 'S':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.S[i];
        break;
    case 'T':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.T[i];
        break;
    case 'U':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.U[i];
        break;
    case 'V':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.V[i];
        break;
    case 'W':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.W[i];
        break;
    case 'X':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.X[i];
        break;
    case 'Y':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.Y[i];
        break;
    case 'Z':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.Z[i];
        break;
    case 's':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.s[i];
        break;
    case 't':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.t[i];
        break;
    case 'k':
        for(i = 0; i < 3; i++)
            subpath_start_string[i] = recip_lattice->vertex.k[i];
        break;
    }

    switch(e_p)
    {
    case 'A':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.A[i];
        break;
    case 'B':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.B[i];
        break;
    case 'C':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.C[i];
        break;
    case 'D':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.D[i];
        break;
    case 'E':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.E[i];
        break;
    case 'F':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.F[i];
        break;
    case 'G':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.G[i];
        break;
    case 'H':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.H[i];
        break;
    case 'I':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.I[i];
        break;
    case 'J':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.J[i];
        break;
    case 'K':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.K[i];
        break;
    case 'L':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.L[i];
        break;
    case 'M':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.M[i];
        break;
    case 'N':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.N[i];
        break;
    case 'O':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.O[i];
        break;
    case 'P':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.P[i];
        break;
    case 'Q':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.Q[i];
        break;
    case 'R':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.R[i];
        break;
    case 'S':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.S[i];
        break;
    case 'T':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.T[i];
        break;
    case 'U':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.U[i];
        break;
    case 'V':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.V[i];
        break;
    case 'W':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.W[i];
        break;
    case 'X':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.X[i];
        break;
    case 'Y':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.Y[i];
        break;
    case 'Z':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.Z[i];
        break;
    case 's':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.s[i];
        break;
    case 't':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.t[i];
        break;
    case 'k':
        for(i = 0; i < 3; i++)
            subpath_end_string[i] = recip_lattice->vertex.k[i];
        break;
    }
}