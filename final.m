

clear all
close all
clc

vertical=true;
horizontal=false;
OutOfFocus=false;
motion=4;%amount of camera motion
focus_strength=4;

% number of rows of your reference image. Use a small number if your computer is slow
row = 30;
img_ori = im2double(imresize(rgb2gray(imread('img_test_lr.png')), [row NaN]));
figure
imshow(img_ori)
title('Reference Image')

sz=size(img_ori);

Alen=(sz(1)*sz(2));
A=single(zeros(Alen));

if horizontal 
    len = round(0.1*motion/3 * size(img_ori,1));
    for i = 1:size(img_ori,1)
    for j = 1:size(img_ori,2)-len+1
        img_motion(i, j) = mean(img_ori(i, j:j+len-1));
    end
end
figure
imshow(img_motion)
title('Horizontal Blur Image')


motion_matrix = zeros(numel(img_motion), numel(img_ori));
order = reshape((1:numel(img_motion))', size(img_motion));
for i = 1:size(img_ori,1)
    fprintf('building motion matrix...%.1f%%\n', i/size(img_ori,1)*100);
    for j = 1:size(img_ori,2)-len+1
        img_temp = zeros(size(img_ori));
        img_temp(i, j:j+len-1) = 1.0/len;
        motion_matrix(order(i, j), :) = reshape(img_temp, 1, []);
    end
end


img_matrix_blur = reshape(motion_matrix * reshape(img_ori, [], 1), size(img_motion));
if max(max(abs(img_matrix_blur - img_motion))) > 1e-10
    error('wrong motion matrix');
end


boundary_matrix = zeros((len-1)*size(img_ori,1), numel(img_ori));
boundary_counter = 1;
for i = 1:size(img_ori,1)
    fprintf('building boundary matrix...%.1f%%\n', i/size(img_ori,1)*100);
    for j = 1:len-1
        img_temp = zeros(size(img_ori));
        img_temp(i, j) = 1.0;
        boundary_matrix(boundary_counter, :) = reshape(img_temp, 1, []);
        boundary_vector(boundary_counter, 1) = img_ori(i, j);
        boundary_counter = boundary_counter + 1;
    end
end

% linear equations: Ax = b
A = [motion_matrix; boundary_matrix];
b = [reshape(img_motion, [], 1); boundary_vector];

[result]=SolveLUD(A, b);
sizeR=size(result);
img_deblur = reshape(result, size(img_ori));


if max(max(abs(img_deblur - img_ori))) > 1e-10
    error('wrong deblur image');
end
figure
imshow(img_deblur)
title('Deblur Image')
disp('Done!')

end

if vertical
    for i=1:Alen
       for j=1:motion
           current_j=j-floor(motion/2); 
           if i+current_j>0 && i+current_j<=Alen
               A(i+current_j,i)=1/motion;           %1/number of elements plced vertically along horizontal
           end
       end

    end
    figure

lin_img=[];
goosdji=img_ori(1,:);

lin_img=reshape(img_ori,[ Alen,  1]);
blur_lin=A*lin_img;
blur_img=reshape(blur_lin,[ sz(1), sz(2)]);
figure
imshow(blur_img);
title('Vertical Blur Image')


x=SolveLUD(A,blur_lin);

good_img=reshape(x,[ sz(1), sz(2)]);
figure
imshow(good_img);
title('Deblur Image')
disp('Done!')
end

if OutOfFocus
    for i=1:Alen
       for j=1:motion
           current_j=j-floor(motion/2);
           if (i+current_j>0 && i+current_j<=Alen)&&((i+current_j)~=i)
               A(i+current_j,i)=1/(focus_strength+4);           %values placed adjacent, with focus stength concentrated along horizontal
              
           end
           
           
           
           
           
       end
     A(i,i)=focus_strength/(focus_strength+4);
    end
    
    
    figure

lin_img=[];
goosdji=img_ori(1,:);

lin_img=reshape(img_ori,[ Alen,  1]);
blur_lin=A*lin_img;
blur_img=reshape(blur_lin,[ sz(1), sz(2)]);
figure
imshow(blur_img);
title('Out of Focus Blur Image')

%kernel error testing
% for i=1:Alen
%     if i+10<=Alen && mod(i,Alen/10)==0
%     A(i+10,i)=A(i,i)+15;
%     end
% end

x=SolveLUD(A,blur_lin);

good_img=reshape(x,[ sz(1), sz(2)]);
figure
imshow(good_img);
title('Deblur Image')
disp('Done!')
end
    








function x = SolveLUD(A, b)
[L U, P]=myLUD(A) ;

[z , ~]=size(A);


%pivot B
    b=P*b;

%  solve 'Ly=b' 
     for j=1:z

        for k=1:j-1
           b(j)=b(j)-L(j,k)*b(k);   %current element is sum of previous, back multiplied with L matrix
        end
        b(j) =b(j)/L(j,j); 
     end

    % use y to solve Rx = y 
    x = zeros( z, 1 );

    for i=z:-1:1    
     x(i) = ( b(i) - U(i, :)*x )/U(i, i); %current element is difference of previous, forward multiplied with ratio of U matrix
    end

x

end

function [L,U,P] = myLUD(X)
sz=size(X);
if sz(1)~=sz(2)
   error("Matrix not square") 
end
n=sz(1);
augP=[X eye(sz)];

for k=1:n-1
    %partial pivoting init
    [~, Pk]=max(abs(augP(k:n,k)));
    
    Pm=k+(Pk-1);
    if Pk>1
        augP([k Pm],:)=augP([Pm k],:); %swap rows
    end
    %decomp
    for m=k+1:n
        augP(m,k) = augP(m,k)/augP(k,k); 
        augP(m,k+1:n) = augP(m,k + 1:n)-augP(m,k)*augP(k,k + 1:n);
    end
end
P = augP(1:n, n + 1:n + n); %augmented permutation matrix for partial pivot
U=zeros(n);
L=zeros(n);

for m = 1:n
    for i = 1:n
        if m == i
            L(m,m) = 1.; U(m,m) = augP(m,m);        %Edge cases handled
        elseif m > i
            L(m,i) = augP(m,i); U(m,i) = 0.; %L is derived from pivot elements
        else
            L(m,n) = 0.;                        
            U(m,i) = augP(m,i);     %U is derived from pivot elements
        end
    end
end

end
