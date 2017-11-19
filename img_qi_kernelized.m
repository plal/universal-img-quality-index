function quality = img_qi(img1, img2, kernelsize)

N = kernelsize.^2;
sum2_filter = ones(kernelsize);

img1_sq = img1.*img1;
img2_sq = img2.*img2;
img12 = img1.*img2;

img1_sum   = filter2(sum2_filter, img1, 'valid');
img2_sum   = filter2(sum2_filter, img2, 'valid');
img1_sq_sum = filter2(sum2_filter, img1_sq, 'valid');
img2_sq_sum = filter2(sum2_filter, img2_sq, 'valid');
img12_sum = filter2(sum2_filter, img12, 'valid');

img12_sum_mul = img1_sum.*img2_sum;
img12_sq_sum_mul = img1_sum.*img1_sum + img2_sum.*img2_sum;
top = 4*(N*img12_sum - img12_sum_mul).*img12_sum_mul;
bot = (N*(img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul).*img12_sq_sum_mul;

quality = mean(mean(top./bot));