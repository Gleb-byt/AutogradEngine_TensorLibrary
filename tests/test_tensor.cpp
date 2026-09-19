#include <gtest/gtest.h>
#include "../include/tensor.hpp"

TEST(TensorTest,Addition) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    Tensor c = a + b;

    EXPECT_FLOAT_EQ(c[0], 6.0f);
    EXPECT_FLOAT_EQ(c[1], 8.0f);
    EXPECT_FLOAT_EQ(c[2], 10.0f);
    EXPECT_FLOAT_EQ(c[3], 12.0f);
}



TEST(TensorTest, Matmul) {
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({5,6,7,8}, {2,2});

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape_[0],2);
    EXPECT_EQ(c.shape_[1], 2);

    EXPECT_EQ(c[0], 19);
    EXPECT_EQ(c[1], 22);
    EXPECT_EQ(c[2], 43);
    EXPECT_EQ(c[3], 50);

}


TEST(TensorThrows, MatmulIncompatibleShapes) {
    Tensor a({1, 2, 3, 4}, {1, 4});
    Tensor b({1, 2, 3, 4}, {2, 2});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);

    Tensor a({1,2,3,4,5,6}, {1,2,3});
    Tensor b({7,8,9,10,11,12}, {3,2,1});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);

    Tensor a({1,2,3,4}, {4});
    Tensor b({1,2}, {2});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);


}

