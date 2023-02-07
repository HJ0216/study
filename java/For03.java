import java.text.*;

public class For03 {

	public static void main(String[] args) {
		int i, sum=0, mul=1;

		DecimalFormat df = new DecimalFormat();
		
		for(i=1; i<11; i++) {
			sum += i;
			mul *= i;
			
			System.out.println("i: " + i + ", sum: " + sum + ", mul: " + df.format(mul));
		}
		
	}
	
}