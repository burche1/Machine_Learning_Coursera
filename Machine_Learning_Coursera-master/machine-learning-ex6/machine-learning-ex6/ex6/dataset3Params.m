function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];   %values that we shoul try

error = zeros(length(steps),length(steps));     %matrix of n_steps x n_steps
for i=1:length(steps);                          %first for to increment C value
    C = steps(i);
    for j=1:length(steps);                      %second for to increment sigma at the C value
        sigma = steps(j);    
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));  %training a model with X and y
        predictions = svmPredict(model, Xval);  %generate predictions to Xval
        error(i,j) = mean(double(predictions ~= yval));     %generate errors with vector predictions and yval (true labels of cross validation set)
    end
end
[r,c] = find(error==min(error(:)));             %position of the minimum error on the matrix

C = steps(r);                                   %line position is the C set on the first for
sigma = steps(c);                               %collum position is the sigma set on the second for

% =========================================================================

end
