function average_vp = fitVanishingPoints(lines)
% Given a set of lines which are parallel in the real word, in the form of
% lines=[l1;l2;l3;], compute all the possible intersection among them and
% return the average intersection point

vp = [];

for i = 1:size(lines, 1)
    for j = i+1:size(lines, 1)
        v = cross(lines(i,:), lines(j,:));
        v = v./v(3);
        vp = [vp; v];
    end
end

average_vp = sum(vp, 1)/size(vp,1);

end