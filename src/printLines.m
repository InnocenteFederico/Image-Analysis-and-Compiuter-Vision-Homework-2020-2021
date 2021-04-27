function printLines(lines, image, R)
% Plot all the lines "lines" on an image "image" with system reference "R"
% The lines are of the form lines=[l1;l2;l3,;l4;...]

n = size(lines);
n = n(1);

for i = 1:n
    lines(i,:) = lines(i,:)./lines(i,3);
end

x = [-10000, +10000];

imshow(image, R), hold on
for i = 1:n
    a = lines(i,1);
    b = lines(i,2);
    c = lines(i,3);
    plot(x, [(a*x(1) + c) / -b, (a*x(2) + c) / -b],'LineWidth', 2,'Color','red');
end
hold off;

end