function [class,d]=museClassify(z,X)
s = size(X);
n=s(1);
class=1;
v = X(1,:)-z;
d = v*v';

for(i=2:1:n)
    v = X(i,:)-z;
    d1 = v*v';
    if (d1 < d)
        class=i;
        d=d1;
    end
end
return
end

    