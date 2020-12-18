#include "FAME_Internal_Common.h"
// 2020-02-19

void MV(realCPU* point, realCPU* M, realCPU V1, realCPU V2, realCPU V3)
{
    point[0] = M[0] * V1 + M[3] * V2 + M[6] * V3;
    point[1] = M[1] * V1 + M[4] * V2 + M[7] * V3;
    point[2] = M[2] * V1 + M[5] * V2 + M[8] * V3;
}

int FAME_Parameter_Brillouin_Zone_Point(VERTEX* Vertex, LATTICE Lattice, realCPU* reciprocal_lattice_vector_b)
{
    realCPU a     = Lattice.lattice_constant.a;
    realCPU b     = Lattice.lattice_constant.b;
    realCPU c     = Lattice.lattice_constant.c;
    realCPU alpha = Lattice.lattice_constant.alpha;
//    realCPU beta  = Lattice.lattice_constant.beta;
//    realCPU gamma = Lattice.lattice_constant.gamma;

    realCPU eta, varsigma, delta, phi, mu, temp, nu, psi, rho, omega;

    if(strcmp(Lattice.lattice_type, "simple_cubic") == 0)
    {
        MV(Vertex->G, reciprocal_lattice_vector_b, 0.0, 0.0, 0.0);
        MV(Vertex->M, reciprocal_lattice_vector_b, 0.5, 0.5, 0.0);
        MV(Vertex->R, reciprocal_lattice_vector_b, 0.5, 0.5, 0.5);
        MV(Vertex->X, reciprocal_lattice_vector_b, 0.0, 0.5, 0.0);
    }
    else if(strcmp(Lattice.lattice_type, "face_centered_cubic") == 0)
    {
        MV(Vertex->G, reciprocal_lattice_vector_b, 0.000, 0.000, 0.000);
        MV(Vertex->K, reciprocal_lattice_vector_b, 0.375, 0.375, 0.750);
        MV(Vertex->L, reciprocal_lattice_vector_b, 0.500, 0.500, 0.500);
        MV(Vertex->U, reciprocal_lattice_vector_b, 0.625, 0.250, 0.625);
        MV(Vertex->W, reciprocal_lattice_vector_b, 0.500, 0.250, 0.750);
        MV(Vertex->X, reciprocal_lattice_vector_b, 0.500, 0.000, 0.500);
    }
    else if(strcmp(Lattice.lattice_type, "body_centered_cubic") == 0)
    {
        MV(Vertex->G, reciprocal_lattice_vector_b, 0.00,  0.00, 0.00);
        MV(Vertex->N, reciprocal_lattice_vector_b, 0.00,  0.00, 0.50);
        MV(Vertex->H, reciprocal_lattice_vector_b, 0.50, -0.50, 0.50);
        MV(Vertex->P, reciprocal_lattice_vector_b, 0.25,  0.25, 0.25);
    }
    else if(strcmp(Lattice.lattice_type, "primitive_tetragonal") == 0)
    {
        MV(Vertex->G, reciprocal_lattice_vector_b, 0.0, 0.0, 0.0);
        MV(Vertex->A, reciprocal_lattice_vector_b, 0.5, 0.5, 0.5);
        MV(Vertex->M, reciprocal_lattice_vector_b, 0.5, 0.5, 0.0);
        MV(Vertex->R, reciprocal_lattice_vector_b, 0.0, 0.5, 0.5);
        MV(Vertex->X, reciprocal_lattice_vector_b, 0.0, 0.5, 0.0);
        MV(Vertex->Z, reciprocal_lattice_vector_b, 0.0, 0.0, 0.5);
    }
    else if(strcmp(Lattice.lattice_type, "body_centered_tetragonal") == 0)
    {
        if(c < a)
        {
            eta      = (1.0 + pow(c / a, 2)) / 4.0;
            MV(Vertex->G, reciprocal_lattice_vector_b,  0.00,      0.00, 0.00);
            MV(Vertex->X, reciprocal_lattice_vector_b,  0.00,      0.00, 0.50);
            MV(Vertex->M, reciprocal_lattice_vector_b, -0.50,      0.50, 0.50);
            MV(Vertex->N, reciprocal_lattice_vector_b,  0.00,      0.50, 0.00);
            MV(Vertex->Z, reciprocal_lattice_vector_b,   eta,       eta, -eta);
            MV(Vertex->B, reciprocal_lattice_vector_b,  -eta, 1.0 - eta,  eta);
            MV(Vertex->P, reciprocal_lattice_vector_b,  0.25,      0.25, 0.25);
        }
        else if(c > a)
        {
            eta      = (1.0 + pow(a / c, 2)) / 4.0;
            varsigma = 0.5 * pow(a, 2) / pow(c, 2);
            MV(Vertex->G, reciprocal_lattice_vector_b,      0.00,      0.00,      0.00);
            MV(Vertex->X, reciprocal_lattice_vector_b,      0.00,      0.00,      0.50);
            MV(Vertex->Y, reciprocal_lattice_vector_b, -varsigma,  varsigma,      0.50);
            MV(Vertex->A, reciprocal_lattice_vector_b,      0.50,      0.50, -varsigma);
            MV(Vertex->s, reciprocal_lattice_vector_b,      -eta,       eta,       eta);
            MV(Vertex->t, reciprocal_lattice_vector_b,       eta, 1.0 - eta,      -eta);
            MV(Vertex->N, reciprocal_lattice_vector_b,      0.00,      0.50,      0.00);
            MV(Vertex->P, reciprocal_lattice_vector_b,      0.25,      0.25,      0.25);
            MV(Vertex->Z, reciprocal_lattice_vector_b,      0.50,      0.50,     -0.50);
        }
        else
            printf("Error, Lattice_constant.a = Lattice_constant.c\n");
    }
    else if(strcmp(Lattice.lattice_type, "primitive_orthorhombic") == 0)
    {
        MV(Vertex->G, reciprocal_lattice_vector_b, 0.0, 0.0, 0.0);
        MV(Vertex->R, reciprocal_lattice_vector_b, 0.5, 0.5, 0.5);
        MV(Vertex->S, reciprocal_lattice_vector_b, 0.5, 0.5, 0.0);
        MV(Vertex->T, reciprocal_lattice_vector_b, 0.0, 0.5, 0.5);
        MV(Vertex->U, reciprocal_lattice_vector_b, 0.5, 0.0, 0.5);
        MV(Vertex->X, reciprocal_lattice_vector_b, 0.5, 0.0, 0.0);
        MV(Vertex->Y, reciprocal_lattice_vector_b, 0.0, 0.5, 0.0);
        MV(Vertex->Z, reciprocal_lattice_vector_b, 0.0, 0.0, 0.5);
    }
    else if(strcmp(Lattice.lattice_type, "face_centered_orthorhombic") == 0)
    {
        if((1.0 / pow(a, 2)) >= (1.0 / pow(b, 2) + 1.0 / pow(c, 2)))
        {
            eta      = (1.0 + pow(a / b, 2) + pow(a / c, 2)) / 4.0;
            varsigma = (1.0 + pow(a / b, 2) - pow(a / c, 2)) / 4.0;
            MV(Vertex->G, reciprocal_lattice_vector_b, 0.0,            0.0,            0.0);
            MV(Vertex->L, reciprocal_lattice_vector_b, 0.5,            0.5,            0.5);
            MV(Vertex->A, reciprocal_lattice_vector_b, 0.5, 0.5 + varsigma,       varsigma);
            MV(Vertex->B, reciprocal_lattice_vector_b, 0.5, 0.5 - varsigma, 1.0 - varsigma);
            MV(Vertex->T, reciprocal_lattice_vector_b, 1.0,            0.5,            0.5);
            MV(Vertex->Y, reciprocal_lattice_vector_b, 0.5,            0.0,            0.5);
            MV(Vertex->X, reciprocal_lattice_vector_b, 0.0,            eta,            eta);
            MV(Vertex->J, reciprocal_lattice_vector_b, 1.0,      1.0 - eta,      1.0 - eta);
            MV(Vertex->Z, reciprocal_lattice_vector_b, 0.5,            0.5,            0.0);
        }
        else
        {
            eta      = (1.0 + pow(a / b, 2) - pow(a / c, 2)) / 4.0;
            delta    = (1.0 + pow(b / a, 2) - pow(b / c, 2)) / 4.0;
            phi      = (1.0 + pow(c / b, 2) - pow(c / a, 2)) / 4.0;
            MV(Vertex->G, reciprocal_lattice_vector_b,         0.0,       0.0,         0.0);
            MV(Vertex->L, reciprocal_lattice_vector_b,         0.5,       0.5,         0.5);
            MV(Vertex->C, reciprocal_lattice_vector_b,         0.5, 0.5 - eta,   1.0 - eta);
            MV(Vertex->E, reciprocal_lattice_vector_b,         0.5, 0.5 + eta,         eta);
            MV(Vertex->D, reciprocal_lattice_vector_b, 0.5 - delta,       0.5, 1.0 - delta);
            MV(Vertex->F, reciprocal_lattice_vector_b, 0.5 + delta,       0.5,       delta);
            MV(Vertex->H, reciprocal_lattice_vector_b,   1.0 - phi, 0.5 - phi,         0.5);
            MV(Vertex->Ii, reciprocal_lattice_vector_b,         phi, 0.5 + phi,         0.5);
            MV(Vertex->X, reciprocal_lattice_vector_b,         0.0,       0.5,         0.5);
            MV(Vertex->Y, reciprocal_lattice_vector_b,         0.5,       0.0,         0.5);
            MV(Vertex->Z, reciprocal_lattice_vector_b,         0.5,       0.5,         0.0);
        }
    }
    else if(strcmp(Lattice.lattice_type, "body_centered_orthorhombic") == 0)
    {
        varsigma = (1.0 + pow(a / c, 2)) / 4.0;
        eta      = (1.0 + pow(b / c, 2)) / 4.0;
        delta    = (pow(b, 2) - pow(a, 2)) / (4.0 * pow(c, 2));
        mu       = (pow(a, 2) + pow(b, 2)) / (4.0 * pow(c, 2));
        MV(Vertex->G, reciprocal_lattice_vector_b,        0.00,           0.00,        0.00);
        MV(Vertex->L, reciprocal_lattice_vector_b,         -mu,             mu, 0.5 - delta);
        MV(Vertex->Ii, reciprocal_lattice_vector_b,          mu,            -mu, 0.5 + delta);
        MV(Vertex->J, reciprocal_lattice_vector_b, 0.5 - delta,    0.5 + delta,         -mu);
        MV(Vertex->R, reciprocal_lattice_vector_b,        0.00,           0.50,        0.00);
        MV(Vertex->S, reciprocal_lattice_vector_b,        0.50,           0.00,        0.00);
        MV(Vertex->T, reciprocal_lattice_vector_b,        0.00,           0.00,        0.50);
        MV(Vertex->W, reciprocal_lattice_vector_b,        0.25,           0.25,        0.25);
        MV(Vertex->X, reciprocal_lattice_vector_b,   -varsigma,       varsigma,    varsigma);
        MV(Vertex->A, reciprocal_lattice_vector_b,    varsigma, 1.0 - varsigma,   -varsigma);
        MV(Vertex->Y, reciprocal_lattice_vector_b,         eta,           -eta,         eta);
        MV(Vertex->B, reciprocal_lattice_vector_b,   1.0 - eta,            eta,        -eta);
        MV(Vertex->Z, reciprocal_lattice_vector_b,        0.50,           0.50,       -0.50);
    }
    else if(strcmp(Lattice.lattice_type, "c_base_centered_orthorhombic") == 0)
    {
        varsigma = (1.0 + pow(a / b, 2)) / 4.0;
        MV(Vertex->G, reciprocal_lattice_vector_b,       0.0,            0.0, 0.0);
        MV(Vertex->R, reciprocal_lattice_vector_b,       0.0,            0.5, 0.5);
        MV(Vertex->S, reciprocal_lattice_vector_b,       0.0,            0.5, 0.0);
        MV(Vertex->T, reciprocal_lattice_vector_b,      -0.5,            0.5, 0.5);
        MV(Vertex->A, reciprocal_lattice_vector_b,  varsigma,       varsigma, 0.5);
        MV(Vertex->B, reciprocal_lattice_vector_b, -varsigma, 1.0 - varsigma, 0.5);
        MV(Vertex->X, reciprocal_lattice_vector_b,  varsigma,       varsigma, 0.0);
        MV(Vertex->C, reciprocal_lattice_vector_b, -varsigma, 1.0 - varsigma, 0.0);
        MV(Vertex->Y, reciprocal_lattice_vector_b,      -0.5,            0.5, 0.0);
        MV(Vertex->Z, reciprocal_lattice_vector_b,       0.0,            0.0, 0.5);
    }
    else if(strcmp(Lattice.lattice_type, "hexagonal") == 0)
    {
        MV(Vertex->G, reciprocal_lattice_vector_b,   0.0,  0.0, 0.0);
        MV(Vertex->A, reciprocal_lattice_vector_b,   0.0,  0.0, 0.5);
        MV(Vertex->H, reciprocal_lattice_vector_b,  1/3., 1/3., 0.5);
        MV(Vertex->K, reciprocal_lattice_vector_b,  1/3., 1/3., 0.0);
        MV(Vertex->L, reciprocal_lattice_vector_b,   0.5,  0.0, 0.5);
        MV(Vertex->M, reciprocal_lattice_vector_b,   0.5,  0.0, 0.0);
        MV(Vertex->k, reciprocal_lattice_vector_b,  (0.5 * sqrt(2) - 1/3.), -1/3., 0.0);
    }
    else if(strcmp(Lattice.lattice_type, "rhombohedral") == 0)
    {
        if(Lattice.lattice_constant.theta_1 < pi / 2)
        {
            eta      = (1.0 + 4.0 * cos(Lattice.lattice_constant.theta_1)) / (2.0 + 4.0 * cos(Lattice.lattice_constant.theta_1));
            nu       = 3.0 / 4.0 - eta / 2.0;
            MV(Vertex->G, reciprocal_lattice_vector_b,      0.0,       0.0,       0.0);
            MV(Vertex->F, reciprocal_lattice_vector_b,      0.5,       0.5,       0.0);
            MV(Vertex->B, reciprocal_lattice_vector_b,      eta,       0.5, 1.0 - eta);
            MV(Vertex->C, reciprocal_lattice_vector_b,      0.5, 1.0 - eta, eta - 1.0);
            MV(Vertex->L, reciprocal_lattice_vector_b,      0.5,       0.0,       0.0);
            MV(Vertex->Ii, reciprocal_lattice_vector_b,      0.0,       0.0,      -0.5);
            MV(Vertex->P, reciprocal_lattice_vector_b,      eta,        nu,        nu);
            MV(Vertex->R, reciprocal_lattice_vector_b, 1.0 - nu,  1.0 - nu, 1.0 - eta);
            MV(Vertex->S, reciprocal_lattice_vector_b,       nu,        nu, eta - 1.0);
            MV(Vertex->Q, reciprocal_lattice_vector_b, 1.0 - nu,        nu,       0.0);
            MV(Vertex->X, reciprocal_lattice_vector_b,       nu,       0.0,       -nu);
            MV(Vertex->Z, reciprocal_lattice_vector_b,      0.5,       0.5,       0.5);
        }
        else if(Lattice.lattice_constant.theta_1 > pi / 2)
        {
            eta      = 1.0 / (2.0 * pow(tan(Lattice.lattice_constant.theta_1 / 2), 2));
            nu       = 3.0 / 4.0 - eta / 2.0;
            MV(Vertex->G, reciprocal_lattice_vector_b,       0.0,      0.0,      0.0);
            MV(Vertex->F, reciprocal_lattice_vector_b,       0.5,     -0.5,      0.0);
            MV(Vertex->L, reciprocal_lattice_vector_b,       0.5,      0.0,      0.0);
            MV(Vertex->Z, reciprocal_lattice_vector_b,       0.5,     -0.5,      0.5);
            MV(Vertex->P, reciprocal_lattice_vector_b,  1.0 - nu,      -nu, 1.0 - nu);
            MV(Vertex->R, reciprocal_lattice_vector_b,        nu, nu - 1.0, nu - 1.0);
            MV(Vertex->Q, reciprocal_lattice_vector_b,       eta,      eta,      eta);
            MV(Vertex->S, reciprocal_lattice_vector_b, 1.0 - eta,     -eta,     -eta);
        }
    }
    else if(strcmp(Lattice.lattice_type, "primitive_monoclinic") == 0)
    {
        eta      = (1.0 - b * (cos(alpha) / c)) / (2.0 * pow(sin(alpha), 2));
        nu       = 1.0 / 2.0 - eta * c * (cos(alpha) / b);
        MV(Vertex->G, reciprocal_lattice_vector_b, 0.0,       0.0,      0.0);
        MV(Vertex->A, reciprocal_lattice_vector_b, 0.5,       0.5,      0.0);
        MV(Vertex->C, reciprocal_lattice_vector_b, 0.0,       0.5,      0.5);
        MV(Vertex->D, reciprocal_lattice_vector_b, 0.5,       0.0,      0.5);
        MV(Vertex->F, reciprocal_lattice_vector_b, 0.5,       0.0,     -0.5);
        MV(Vertex->E, reciprocal_lattice_vector_b, 0.5,       0.5,      0.5);
        MV(Vertex->H, reciprocal_lattice_vector_b, 0.0,       eta, 1.0 - nu);
        MV(Vertex->Ii, reciprocal_lattice_vector_b, 0.0, 1.0 - eta,       nu);
        MV(Vertex->J, reciprocal_lattice_vector_b, 0.0,       eta,      -nu);
        MV(Vertex->M, reciprocal_lattice_vector_b, 0.5,       eta, 1.0 - nu);
        MV(Vertex->N, reciprocal_lattice_vector_b, 0.5, 1.0 - eta,       nu);
        MV(Vertex->O, reciprocal_lattice_vector_b, 0.5,       eta,      -nu);
        MV(Vertex->X, reciprocal_lattice_vector_b, 0.0,       0.5,      0.0);
        MV(Vertex->Y, reciprocal_lattice_vector_b, 0.0,       0.0,      0.5);
        MV(Vertex->B, reciprocal_lattice_vector_b, 0.0,       0.0,     -0.5);
        MV(Vertex->Z, reciprocal_lattice_vector_b, 0.5,       0.0,      0.0);
    }
    else if(strcmp(Lattice.lattice_type, "base_centered_monoclinic") == 0)
    {
        temp = reciprocal_lattice_vector_b[0] * reciprocal_lattice_vector_b[3] +\
               reciprocal_lattice_vector_b[1] * reciprocal_lattice_vector_b[4] +\
               reciprocal_lattice_vector_b[2] * reciprocal_lattice_vector_b[5];
        if(temp <= 0)
        {
            varsigma = (2 - b * cos(alpha) / c)/(4 * pow(sin(alpha), 2));
            eta      = 1/2 + 2 * varsigma * c * cos(alpha) / b;
            psi      = 3/4 - pow(a, 2) / (4 * pow(b, 2) * pow(sin(alpha), 2));
            phi      = psi + (3/4 - psi) * b * cos(alpha) / c;
            MV(Vertex->G, reciprocal_lattice_vector_b,            0.0,            0.0,       0.0);
            MV(Vertex->N, reciprocal_lattice_vector_b,            0.5,            0.0,       0.0);
            MV(Vertex->S, reciprocal_lattice_vector_b,            0.0,           -0.5,       0.0);
            MV(Vertex->F, reciprocal_lattice_vector_b, 1.0 - varsigma, 1.0 - varsigma, 1.0 - eta);
            MV(Vertex->J, reciprocal_lattice_vector_b,       varsigma,       varsigma,       eta);
            MV(Vertex->K, reciprocal_lattice_vector_b,      -varsigma,      -varsigma, 1.0 - eta);
            MV(Vertex->O, reciprocal_lattice_vector_b, 1.0 - varsigma,      -varsigma, 1.0 - eta);
            MV(Vertex->Ii, reciprocal_lattice_vector_b,            phi,      1.0 - phi,       0.5);
            MV(Vertex->R, reciprocal_lattice_vector_b,      1.0 - phi,      phi - 1.0,       0.5);
            MV(Vertex->L, reciprocal_lattice_vector_b,            0.5,            0.5,       0.5);
            MV(Vertex->M, reciprocal_lattice_vector_b,            0.5,            0.0,       0.5);
            MV(Vertex->X, reciprocal_lattice_vector_b,      1.0 - psi,      psi - 1.0,       0.0);
            MV(Vertex->T, reciprocal_lattice_vector_b,            psi,      1.0 - psi,       0.0);
            MV(Vertex->U, reciprocal_lattice_vector_b,      psi - 1.0,           -psi,       0.0);
            MV(Vertex->Y, reciprocal_lattice_vector_b,            0.5,            0.5,       0.0);
            MV(Vertex->V, reciprocal_lattice_vector_b,           -0.5,           -0.5,       0.0);
            MV(Vertex->Z, reciprocal_lattice_vector_b,            0.0,            0.0,       0.5);
        }
        else
        {
            temp = b * cos(alpha) / c + pow(b, 2) * pow(sin(alpha), 2) / pow(a, 2);
            if(temp <= 1.0)
            {
                mu       = (1.0 + pow(b / a, 2)) / 4.0;
                delta    = b * c * cos(alpha) / (2.0 * pow(a, 2));
                varsigma = mu - 1.0 / 4.0 + (1.0 - b * cos(alpha) / c) / (4.0 * pow(sin(alpha), 2));
                eta      = 1.0 / 2.0 + 2.0 * varsigma * c * cos(alpha) / b;
                phi      = 1.0 + varsigma - 2.0 * mu;
                psi      = eta - 2.0 * delta;
                MV(Vertex->G, reciprocal_lattice_vector_b,            0.0,       0.0,       0.0);
                MV(Vertex->F, reciprocal_lattice_vector_b,      1.0 - phi, 1.0 - phi, 1.0 - psi);
                MV(Vertex->J, reciprocal_lattice_vector_b,            phi, phi - 1.0,       psi);
                MV(Vertex->K, reciprocal_lattice_vector_b,      1.0 - phi,      -phi, 1.0 - psi);
                MV(Vertex->H, reciprocal_lattice_vector_b,       varsigma,  varsigma,       eta);
                MV(Vertex->P, reciprocal_lattice_vector_b, 1.0 - varsigma, -varsigma, 1.0 - eta);
                MV(Vertex->Q, reciprocal_lattice_vector_b,      -varsigma, -varsigma, 1.0 - eta);
                MV(Vertex->Ii, reciprocal_lattice_vector_b,            0.5,      -0.5,       0.5);
                MV(Vertex->M, reciprocal_lattice_vector_b,            0.5,       0.0,       0.5);
                MV(Vertex->N, reciprocal_lattice_vector_b,            0.5,       0.0,       0.0);
                MV(Vertex->S, reciprocal_lattice_vector_b,            0.0,      -0.5,       0.0);
                MV(Vertex->X, reciprocal_lattice_vector_b,            0.5,      -0.5,       0.0);
                MV(Vertex->Y, reciprocal_lattice_vector_b,             mu,        mu,     delta);
                MV(Vertex->V, reciprocal_lattice_vector_b,       1.0 - mu,       -mu,    -delta);
                MV(Vertex->W, reciprocal_lattice_vector_b,            -mu,       -mu,    -delta);
                MV(Vertex->A, reciprocal_lattice_vector_b,             mu,  mu - 1.0,     delta);
                MV(Vertex->Z, reciprocal_lattice_vector_b,            0.0,       0.0,       0.5);
            }
            else
            {
                varsigma = (pow(b / a, 2) + (1.0 - b * cos(alpha) / c) / pow(sin(alpha), 2)) / 4.0;
                eta      = 1.0 / 2.0 + 2.0 * varsigma * c * cos(alpha) / b;
                mu       = eta / 2.0 + pow(b, 2) / (4.0 * pow(a, 2)) - b * c * cos(alpha) / (2.0 * pow(a, 2));
                nu       = 2.0 * mu - varsigma;
                omega    = ((4.0 * nu -1.0 - pow(b, 2) * pow(sin(alpha), 2) / pow(a, 2)) * c) / (2.0 * b * cos(alpha));
                delta    = varsigma * c * cos(alpha) / b + omega / 2.0 - 1.0 / 4.0;
                rho      = 1.0 - varsigma * pow(a / b, 2);
                MV(Vertex->G, reciprocal_lattice_vector_b,            0.0,       0.0,         0.0);
                MV(Vertex->F, reciprocal_lattice_vector_b,             nu,        nu,       omega);
                MV(Vertex->J, reciprocal_lattice_vector_b,       1.0 - nu,  1.0 - nu, 1.0 - omega);
                MV(Vertex->K, reciprocal_lattice_vector_b,             nu,    nu - 1,       omega);
                MV(Vertex->H, reciprocal_lattice_vector_b,       varsigma,  varsigma,         eta);
                MV(Vertex->P, reciprocal_lattice_vector_b, 1.0 - varsigma, -varsigma,   1.0 - eta);
                MV(Vertex->Q, reciprocal_lattice_vector_b,      -varsigma, -varsigma,   1.0 - eta);
                MV(Vertex->Ii, reciprocal_lattice_vector_b,            rho, 1.0 - rho,         0.5);
                MV(Vertex->R, reciprocal_lattice_vector_b,      1.0 - rho, rho - 1.0,         0.5);
                MV(Vertex->L, reciprocal_lattice_vector_b,            0.5,       0.5,         0.5);
                MV(Vertex->M, reciprocal_lattice_vector_b,            0.5,       0.0,         0.5);
                MV(Vertex->N, reciprocal_lattice_vector_b,            0.5,       0.0,         0.0);
                MV(Vertex->S, reciprocal_lattice_vector_b,            0.0,      -0.5,         0.0);
                MV(Vertex->X, reciprocal_lattice_vector_b,            0.5,      -0.5,         0.0);
                MV(Vertex->Y, reciprocal_lattice_vector_b,             mu,        mu,       delta);
                MV(Vertex->V, reciprocal_lattice_vector_b,       1.0 - mu,       -mu,      -delta);
                MV(Vertex->W, reciprocal_lattice_vector_b,            -mu,       -mu,      -delta);
                MV(Vertex->A, reciprocal_lattice_vector_b,             mu,  mu - 1.0,       delta);
                MV(Vertex->Z, reciprocal_lattice_vector_b,            0.0,       0.0,         0.5);
            }
        }
    }
    else if(strcmp(Lattice.lattice_type, "triclinic") == 0)
    {
        temp = reciprocal_lattice_vector_b[3] * reciprocal_lattice_vector_b[6] +\
               reciprocal_lattice_vector_b[4] * reciprocal_lattice_vector_b[7] +\
               reciprocal_lattice_vector_b[5] * reciprocal_lattice_vector_b[8];
        if(temp < 0)
        {
            MV(Vertex->G, reciprocal_lattice_vector_b, 0.0, 0.0, 0.0);
            MV(Vertex->L, reciprocal_lattice_vector_b, 0.5, 0.5, 0.0);
            MV(Vertex->M, reciprocal_lattice_vector_b, 0.0, 0.5, 0.5);
            MV(Vertex->N, reciprocal_lattice_vector_b, 0.5, 0.0, 0.5);
            MV(Vertex->R, reciprocal_lattice_vector_b, 0.5, 0.5, 0.5);
            MV(Vertex->X, reciprocal_lattice_vector_b, 0.5, 0.0, 0.0);
            MV(Vertex->Y, reciprocal_lattice_vector_b, 0.0, 0.5, 0.0);
            MV(Vertex->Z, reciprocal_lattice_vector_b, 0.0, 0.0, 0.5);
        }
        else if(temp > 0)
        {
            MV(Vertex->G, reciprocal_lattice_vector_b,  0.0,  0.0, 0.0);
            MV(Vertex->L, reciprocal_lattice_vector_b,  0.5, -0.5, 0.0);
            MV(Vertex->M, reciprocal_lattice_vector_b,  0.0,  0.0, 0.5);
            MV(Vertex->N, reciprocal_lattice_vector_b, -0.5, -0.5, 0.5);
            MV(Vertex->R, reciprocal_lattice_vector_b,  0.0, -0.5, 0.5);
            MV(Vertex->X, reciprocal_lattice_vector_b,  0.0, -0.5, 0.0);
            MV(Vertex->Y, reciprocal_lattice_vector_b,  0.5,  0.0, 0.0);
            MV(Vertex->Z, reciprocal_lattice_vector_b, -0.5,  0.0, 0.5);
        }
    }

    return 0;
}