function error = mse(img1, img2)
error = mean(mean((img1-img2).^2))
