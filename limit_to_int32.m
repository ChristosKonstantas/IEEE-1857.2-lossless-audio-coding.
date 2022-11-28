function c = limit_to_int32(num)
 if num > 2^31
  c=2^31; 
 elseif num< -2^31 
  c=-2^31; 
 else 
  c=int32(num); 
end

