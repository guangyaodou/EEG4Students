function [E,C] = museClassifyAll(Y,k,X)
  s = size(Y);
  n = s(1);
  for(i1=1:1:n)
      C(i1) = museClassify(Y(i1,:),X);
  end
  D = ones(1,n);
  E = zeros(1,s(2));
  for(i1=1:1:n)
      D(C(i1)) = D(C(i1))+1;
      if (i1>k)
          D(C(i1-k)) = D(C(i1-k))-1;
      end
    
      [m,j]=max(D);
      E(i1)=j;
  end
  return
end

      