%% 
% Homework of Image Analysis and Computer Vision
% Accandemic year 2020/2021
% Federico Innocente

%% Image upload and preprocessing

original_image = imread('Image - Castello di Miramare.jpg');

grayscale_image = rgb2gray(original_image);

% Increase the ontrast of the image
contrast_image = adapthisteq(grayscale_image);

image_size = size(original_image);

% Set the constants to normalize the points
NORM_FACTOR = max(size(grayscale_image));
DENORM_MAT = diag([NORM_FACTOR, NORM_FACTOR,1]);

image_ratio = image_size(1)/image_size(2);

% Set the normalized system reference
RA = imref2d(size(grayscale_image), [0 1], [0 image_ratio]);

%% Horizontal edge detection

% Use canny to detect the edges
edges = edge(contrast_image, 'canny', [0.15 0.25], 1.8);

% Use hough to collect the edges
[Ho,T,R] = hough(edges, 'Theta', [-90:-30 36:89]);
P  = houghpeaks(Ho,100, 'Threshold', 0.1*max(Ho(:)));
lines = houghlines(edges,T,R,P,'FillGap', 40,'MinLength',200);

% Print the result of the hough process
figure(1), imshow(contrast_image), hold on
title("Horizontal lines detected");
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
end

% Homogenize the lines
hom_lines = zeros(length(lines), 3);
for i = 1:length(lines)
    point1 = [lines(i).point1 / NORM_FACTOR, 1];
    point2 = [lines(i).point2 / NORM_FACTOR, 1];
    hom_lines(i,:) = cross(point1, point2);
    hom_lines(i,:) = hom_lines(i,:)./hom_lines(i,3);
end

%% Horizontal edge selection

% Select the chosen lines
% lfn stay for lines parallel to pi on the facade n
% facades 4 and 6 are parallel and so jointed
lf1 = [hom_lines(2,:); hom_lines(36,:); hom_lines(13,:)];
lf2 = [hom_lines(16,:); hom_lines(4,:); hom_lines(42,:); hom_lines(48,:); hom_lines(55,:)];
lf3 = [hom_lines(77,:); hom_lines(17,:); hom_lines(10,:)];
lf5 = [hom_lines(31,:); hom_lines(76,:)];
lf46 = [hom_lines(8,:); hom_lines(18,:); hom_lines(1,:)];

% Plot the selected lines
l = [lf1; lf2; lf3; lf5; lf46];
printLines(l, original_image, RA);

%% Vanishing points computation and line at infinity

% Compute the average vanishing point for each family of parallel lines
vanishing_points = [fitVanishingPoints(lf1); fitVanishingPoints(lf2); fitVanishingPoints(lf3); ... 
    fitVanishingPoints(lf5); fitVanishingPoints(lf46)];

% Compute the line that better fits the vanishing points
linf_coef = polyfit(vanishing_points(:,1), vanishing_points(:,2), 1);
yFitted = polyval(linf_coef, vanishing_points(:,1));
fitted_points = [vanishing_points(:,1), yFitted];
figure(2), imshow(grayscale_image, RA), hold on
title("Image of the line at infinity");
for k = 1:size(vanishing_points, 1)
    plot(fitted_points(k,1), fitted_points(k,2),'g.','MarkerSize',30);
end
line([fitted_points(1,1), fitted_points(5,1)],[fitted_points(1,2),fitted_points(5,2)], 'LineWidth', 2);
hold off;

%% Affine rectification

% Resize the image and the reference system to avoid memory issue on
% computing the trasformation
resized_image = imresize(original_image, 0.4);
grayscale_resized_image = rgb2gray(resized_image);
RA_res = imref2d(size(grayscale_resized_image), [0 1], [0 0.75]);

% Compute the image of the line at infinity
imLinf = [linf_coef(1), -1, linf_coef(2)];
imLinf = imLinf./imLinf(3);

% Perform the affine rectification
H_aff = [1 0 0; 0 1 0; imLinf];
affine_tform = projective2d(H_aff');
[affined_image, RB] = imwarp(resized_image, RA_res, affine_tform);
figure(3), imshow(affined_image, RB);
title("Affined rectified image");

%% Get two affined perpendicoular lines

% Compute two affined perpendicular lines to use them for the metric
% rectification
H = inv(H_aff);
H = H';

l1 = hom_lines(2,:);
l2 = hom_lines(16,:);
l1 = H*l1';
l2 = H*l2';
l1 = l1./l1(3);
l2 = l2./l2(3);

l5 = hom_lines(31,:);
l6 = hom_lines(1,:);
l5 = H*l5';
l6 = H*l6';
l5 = l5./l5(3);
l6 = l6./l6(3);

%% Metric rectification

% Compute the metric rectification matrix
constrains = zeros(2,3);
constrains(1,:) = [l1(1)*l2(1), l1(1)*l2(2)+l1(2)*l2(1), l1(2)*l2(2)];
constrains(2,:) = [l5(1)*l6(1), l5(1)*l6(2)+l5(2)*l6(1), l5(2)*l6(2)];

[~,~,v] = svd(constrains);
s = v(:,end); %[s11,s12,s22];
S = [s(1),s(2); s(2),s(3)];

imDCCP = [S,zeros(2,1); zeros(1,3)]; % the image of the circular points
[U,D,V] = svd(imDCCP);
A = U*sqrt(D)*V';
Ha = eye(3);
Ha(1,1) = A(1,1);
Ha(1,2) = A(1,2);
Ha(2,1) = A(2,1);
Ha(2,2) = A(2,2);

H_rect = inv(Ha);

% Apply the metric rectification on the affined image
metric_tform = projective2d(H_rect');
[metric_image, RC] = imwarp(affined_image, RB, metric_tform);
figure(4), imshow(metric_image, RC);
title("Metric rectified image");

%% Vertical edge detection

% Detect the vertical edges
vertical_edges = edge(contrast_image, 'canny', [0.1 0.2], 2);

% Use hough to collect the relevant edges
[Ho,T,R] = hough(vertical_edges, 'Theta', -15:15);
P  = houghpeaks(Ho,100, 'Threshold', 0.1*max(H(:)));
vertical_lines = houghlines(edges,T,R,P,'FillGap', 35,'MinLength',200);

% Show the result of the hough process
figure(5), imshow(contrast_image), hold on
title("Vertical lines detected");
for k = 1:length(vertical_lines)
   xy = [vertical_lines(k).point1; vertical_lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
end

% Homogenize the detected lines
hom_vertical_lines = zeros(length(vertical_lines), 3);
for i = 1:length(vertical_lines)
    point1 = [vertical_lines(i).point1 / NORM_FACTOR, 1];
    point2 = [vertical_lines(i).point2 / NORM_FACTOR, 1];
    hom_vertical_lines(i,:) = cross(point1, point2);
    hom_vertical_lines(i,:) = hom_vertical_lines(i,:)./hom_vertical_lines(i,3);
end
%% Vertical edge selection

% Select the vertical lines and compute the vertical vanishing point
lv = [hom_vertical_lines(4,:); hom_vertical_lines(1,:); hom_vertical_lines(8,:); ...
    hom_vertical_lines(3,:); hom_vertical_lines(15,:); hom_vertical_lines(14,:); hom_vertical_lines(2,:)];

% Plot selected lines
printLines(lv, original_image, RA);

vpv = fitVanishingPoints(lv);

%% Camera calibration

% Denormalize the vanishing points
% The vaishing points are calculated again, but it would be the same to
% denormalize the ones calculated in the previous steps
vpf1 = fitVanishingPoints(lf1) * DENORM_MAT;
vpf2 = fitVanishingPoints(lf2) * DENORM_MAT;
vpf5 = fitVanishingPoints(lf5) * DENORM_MAT;
vpf46 = fitVanishingPoints(lf46) * DENORM_MAT;
vpv = fitVanishingPoints(lv) * DENORM_MAT;

% Solve the system to get omega
syms w1 w2 w3 w4
w = [w1, 0,  w2; ...
     0,  1,  w3; ...
     w2, w3, w4];
sol = solve( vpf1 * w * vpf2'  == 0, ...
             vpf5 * w * vpf46' == 0, ...
             vpv  * w * vpf2'  == 0, ...
             vpv  * w * vpf1'  == 0); 

w1 = double(sol.w1);
w2 = double(sol.w2);
w3 = double(sol.w3);
w4 = double(sol.w4);

w = [w1, 0, w2; ...
     0, 1 , w3; ...
     w2, w3, w4];

% Apply the Cholesky factorization to omega to compute the calibration
% matrix
K_chol_inv = chol(w);
K = inv(K_chol_inv);
K = K./K(3,3);

%% Camera localization

% Compute the first point on pi as the intersection of lines 1 and 2
plane_origin = cross(lf1(1,:), lf2(1,:));
plane_origin = plane_origin./plane_origin(3);
a = plane_origin;

% Compute other two points on pi as two points on 1 and 2 rispectivly
d = lines(2).point1 / NORM_FACTOR;
d = [d, 1];
b = lines(16).point2 / NORM_FACTOR;
b = [b, 1];

% Get the direction of the two sides already delimited
dir_ad = cross(imLinf, hom_lines(2,:));
dir_ad = dir_ad./dir_ad(3);
dir_ab = cross(imLinf, hom_lines(16,:));
dir_ab = dir_ab./dir_ab(3);

% Compute the lines of the two lines of the rectangula not delimited yet
line_bc = cross(dir_ad, b);
line_bc = line_bc./line_bc(3);
line_dc = cross(dir_ab, d);
line_dc = line_dc./line_dc(3);

% Compute the fourth point as the intersection between the last two lines
c = cross(line_bc, line_dc);
c = c./c(3);

% Apply the affine and the metric transformation to the four points to
% bring them on the rectified plane
a_rec = Ha\H_aff*a'; %H_rect * H_aff * a'
a_rec = a_rec./a_rec(3);
b_rec = Ha\H_aff*b';
b_rec = b_rec./b_rec(3);
c_rec = Ha\H_aff*c';
c_rec = c_rec./c_rec(3);
d_rec = Ha\H_aff*d';
d_rec = d_rec./d_rec(3);

% Compute the lenght of the sides of the rectangula and the ratios
ab_lenght = pdist([a_rec(1:2)'; b_rec(1:2)'],'euclidean');
ad_lenght = pdist([a_rec(1:2)'; d_rec(1:2)'],'euclidean');
cb_lenght = pdist([c_rec(1:2)'; b_rec(1:2)'],'euclidean');
cd_lenght = pdist([c_rec(1:2)'; d_rec(1:2)'],'euclidean');
ratio = ab_lenght / ad_lenght;
ratio2 = ab_lenght / cb_lenght;

% Denormalize the points
a = a * DENORM_MAT;
b = b * DENORM_MAT;
c = c * DENORM_MAT;
d = d * DENORM_MAT;

% Compute the transformation between the world and the plane references
sf = 10;
plane_points = [0,0; 0, sf*ratio; -1*sf,ratio*sf; -1*sf,0];
camera_points = [a(1:2); b(1:2); c(1:2); d(1:2)];
loc_tform = fitgeotrans(plane_points, camera_points, 'projective');
H_loc = loc_tform.T'; 

% Compute the pose of the plane with respect to the camera
%inv(K) * H_loc
pose = K \ H_loc; 

%% Get rotation and transaltion

% Compute the versors of the axis of the plane
i = pose(:, 1) / norm(pose(:, 1), 2);
j = pose(:, 2) / norm(pose(:, 2), 2);
k = cross(i, j) / norm(cross(i, j), 2);
o = pose(:, 3) / norm(pose(:, 1), 2);

% Prepare the rotation matrix and the translation vector
rotation = [i, j, k]';
[U, ~, V] = svd(rotation);
rotation = U * V';
translation = -rotation * o;

%% Plot the localizated camera

% Print the camera localization with respect to the point on the roof of
% facades 1 and 2
plane_points=[0, 0, 0; 0, sf*ratio, 0; sf*-1, sf*ratio, 0; sf*-1, 0, 0];
figure(6);
pcshow(pointCloud(plane_points),'MarkerSize',200), hold on
plotCamera('location',translation,'orientation',rotation','size',1);
title("camera localization w.r.t. 4 points on the roof of facade 1 and 2")
hold off;

%% Facade 1 rectification using the camera calibration matrix

% Compute the line at infinity with respect to the plane on the facade 1
imLinf_f1 = cross(vpv, vpf1);
imLinf_f1 = imLinf_f1./imLinf_f1(3);

% The relationship of this procedure with K is in w, which is w=inv(KK'),
% but since I've already computed w I will keep that result

% Intersect the image of the absolut conic and the the image of the line at
% infinity
syms x y
point = [x, y, 1];
eqs = [imLinf_f1 * point.' == 0, ...
       point * w * point.' == 0 ];
inf_points = solve(eqs);

I1 = [double(inf_points.x(1)), double(inf_points.y(1)), 1];
J1 = [double(inf_points.x(2)), double(inf_points.y(2)), 1];

% Calculate the image of the conic dual to the circular points
Cinf1 = I1.'*J1 + J1.'*I1;

% Compute the projective transformation
[U,D,V] = svd(Cinf1);

D(3,3) = 0;
H_rect_frontal = U*sqrt(D)*V';
H_rect_frontal = H_rect_frontal.*10e+3;
% H_rect_frontal is set to one since otherwise Matlab would produce an
% error on the inverse operation. There was no rescale that lead to a
% possible invertible solution without error. Since this will not affect
% the rectification, I solved the problem by setting the (3,3) element to
% one
H_rect_frontal(3,3) = 1;
H_rect_frontal = inv(H_rect_frontal);

% Apply the projective transformation to the image
frontal_rec_tform = projective2d(H_rect_frontal);
frontal_rec_image = imwarp(original_image, frontal_rec_tform);

% Show the rectified image
figure(7), imshow(frontal_rec_image);
title("Rectification of the facade 1");

