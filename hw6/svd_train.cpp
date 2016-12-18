/*
 * =====================================================================================
 *
 *       Filename:  svd_train.cpp
 *
 *    Description:  SVD SGD training algorithm for ITMO CTD RecSys task
 *
 *        Version:  1.0
 *        Created:  12/16/2016 02:54:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Anton Belyy
 *   Organization:  ITMO CTD
 *
 * =====================================================================================
 */
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

// Flags
const bool READ_FROM_BINARY = true;
const bool INIT_FROM_FILE = !true;

// Parameters
const int N_SVD_FEATURES = 300;
const long long int N_ITERS = 300000000;
const float L_RATE = 0.1;

// Dataset-specific parameters
const int N_ROWS = 100000000;
const int N_USERS = 500000;
const int N_ITEMS = 20000;

// Datatypes
struct df_entry {
    int user;
    int item;
    int rating;
};

struct dataframe {
    int n_rows;
    df_entry rows[N_ROWS];
};

// Huge arrays
dataframe df;
float user_vec[N_USERS][N_SVD_FEATURES];
float item_vec[N_ITEMS][N_SVD_FEATURES];

// Baseline score arrays
#if defined(BASELINE_MODEL)
const float K_MEAN = 2;
float sum_rating[N_ITEMS], g_sum_rating;
float sum_offset[N_USERS], g_sum_offset;
int cnt_rating[N_ITEMS], g_cnt_rating;
int cnt_offset[N_USERS], g_cnt_offset;
#endif

static inline float predict(int user, int item) {
#if defined(BASELINE_MODEL)
    float rating_pred = (g_sum_rating / g_cnt_rating * K_MEAN + sum_rating[item]) / (K_MEAN + cnt_rating[item])
                      + (g_sum_offset / g_cnt_offset * K_MEAN + sum_offset[user]) / (K_MEAN + cnt_offset[user]);
    if (rating_pred < 1) rating_pred = 1;
    if (rating_pred > 5) rating_pred = 5;
    return rating_pred;
#else
    float rating_pred = 0;
    for (int t = 0; t < N_SVD_FEATURES; t++) {
        rating_pred += user_vec[user][t] * item_vec[item][t];
    }
    if (rating_pred < 1) rating_pred = 1;
    if (rating_pred > 5) rating_pred = 5;
    return rating_pred;
#endif
}

int main() {
    // Initialize users & items vectors
    printf("Start initializing...\n");
    if (INIT_FROM_FILE) {
        FILE * fuv = fopen("20161217140707-uv.bin", "rb");
        fread(user_vec, sizeof(user_vec), 1, fuv);
        fclose(fuv);
        FILE * fiv = fopen("20161217140707-iv.bin", "rb");
        fread(item_vec, sizeof(item_vec), 1, fiv);
        fclose(fiv);
    } else {
        for (int i = 0; i < N_SVD_FEATURES; i++) {
            for (int j = 0; j < N_USERS; j++) {
                user_vec[j][i] = 1. / sqrt(N_SVD_FEATURES);
            }
            for (int j = 0; j < N_ITEMS; j++) {
                item_vec[j][i] = 1. / sqrt(N_SVD_FEATURES);
            }
        }
    }
    printf("OK\n");

    // Reading dataset
    printf("Start reading dataset...\n");
    if (READ_FROM_BINARY) {
        FILE * f = fopen("train.bin", "rb");
        fread(&df, sizeof(df), 1, f);
        fclose(f);
    } else {
        FILE * f = fopen("data/learn.ssv", "r");
        fscanf(f, "%*s%*s%*s");
        while (!feof(f)) {
            fscanf(f, "%*d%d%d%d", &df.rows[df.n_rows].user
                                 , &df.rows[df.n_rows].item
                                 , &df.rows[df.n_rows].rating);
            df.n_rows++;
        }
        fclose(f);
    }
    printf("OK\n");

    // Save dataset in binary format
    if (!READ_FROM_BINARY) {
        printf("Dump to file...\n");
        FILE * f = fopen("learn.bin", "wb");
        fwrite(&df, sizeof(char), sizeof(df), f);
        fclose(f);
        printf("OK\n");
    }

    // Training
    printf("Start training...\n");
    clock_t begin = clock();
#if defined(BASELINE_MODEL)
    for (int i = 0; i < df.n_rows; i++) {
        int user = df.rows[i].user, item = df.rows[i].item;
        float rating = (float) df.rows[i].rating;
        sum_rating[item] += rating;
        cnt_rating[item] += 1;
        g_sum_rating += rating;
        g_cnt_rating += 1;
    }
    for (int i = 0; i < df.n_rows; i++) {
        int user = df.rows[i].user, item = df.rows[i].item;
        float rating = (float) df.rows[i].rating;
        float delta_sum_offset = rating - sum_rating[item] / cnt_rating[item];
        sum_offset[user] += delta_sum_offset;
        cnt_offset[user] += 1;
        g_sum_offset += delta_sum_offset;
        g_cnt_offset += 1;
    }
#else
    srand(1);
    for (int i = 0; i < N_SVD_FEATURES; i++) {
        printf("%f\n", N_ITERS / sqrt(i + 1));
        continue;
        for (int j = 0; j < N_ITERS / sqrt(i + 1); j++) {
            int k = rand() % df.n_rows;
            int user = df.rows[k].user, item = df.rows[k].item;
            float rating_true = (float) df.rows[k].rating;

            // Calculate rating using current user / item vectors
            float rating_pred = predict(user, item);

            // Calculate and apply stochastic gradient step
            float err = L_RATE * (rating_true - rating_pred);
            float uv = user_vec[user][i];
            user_vec[user][i] += err * item_vec[item][i];
            item_vec[item][i] += err * uv;
        }
        printf("Finish for feature %d\n", i);
    }
#endif
    clock_t end = clock();
    printf("OK, time elapsed: %.1lfs\n", (double)(end - begin) / CLOCKS_PER_SEC);

    // Holdout MSE
    {
        printf("Start calculating hold-out MSE...\n");
        int n_holdout = 1500000;
        float mse = 0, ninv = 1. / n_holdout;
        FILE * f = fopen("data/holdout.ssv", "r");
        fscanf(f, "%*s%*s%*s");
        for (int i = 0; i < n_holdout; i++) {
            int user, item;
            float rating_true;
            fscanf(f, "%*d%d%d%f", &user, &item, &rating_true);
            float rating_pred = predict(user, item);
            mse += ninv * (rating_true - rating_pred) * (rating_true - rating_pred);
        }
        fclose(f);
        printf("OK, MSE = %.6f\n", mse);
    }

    // Fill out submission
    {
        printf("Start filling out submission...\n");
        FILE * fi = fopen("data/test-ids.csv", "r");
        FILE * fo = fopen("submission-float.csv", "w");
        fscanf(fi, "%*s%*s%*s");
        fprintf(fo, "Id,Prediction\n");
        int i = 1;
        while (!feof(fi)) {
            int id, user, item;
            fscanf(fi, "%d%d%d", &id, &user, &item);
            if (i != id) {
                break;
            }
            float rating_pred = predict(user, item);
            fprintf(fo, "%d,%f\n", id, rating_pred);
            i++;
        }
        fclose(fo);
        fclose(fi);
        printf("OK\n");
    }

    // Save model to file
    {
        char uv_fname[256], iv_fname[256];
        time_t t = time(NULL);
        tm * tmp = localtime(&t);
        printf("Start saving model...\n");
        strftime(uv_fname, sizeof(uv_fname), "%Y%m%d%H%M%S-uv.bin", tmp);
        strftime(iv_fname, sizeof(iv_fname), "%Y%m%d%H%M%S-iv.bin", tmp);
        {
            FILE * f = fopen(uv_fname, "wb");
            fwrite(user_vec, sizeof(char), sizeof(user_vec), f);
            fclose(f);
        }
        {
            FILE * f = fopen(iv_fname, "wb");
            fwrite(item_vec, sizeof(char), sizeof(item_vec), f);
            fclose(f);
        }
        printf("OK\n");
    }

    // Test on manual data
    /*while (!feof(stdin)) {
        int user, item;
        printf("user item:\n");
        scanf("%d%d", &user, &item);
        printf("user vec:");
        for (int t = 0; t < N_SVD_FEATURES; t++) {
            printf(" %.3f", user_vec[user][t]);
        }
        printf("\nitem vec:");
        for (int t = 0; t < N_SVD_FEATURES; t++) {
            printf(" %.3f", item_vec[item][t]);
        }
        printf("\n=> %f\n", predict(user, item));
    }*/

    return 0;
}
