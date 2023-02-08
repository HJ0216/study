import java.text.*;

public class Multifor02 {

	public static void main(String[] args) {
		int dan;
		int i;
		
		DecimalFormat df = new DecimalFormat("00");
		
		for(dan=2; dan<10; dan++) {
			for(i=1; i<10; i++) {
				System.out.print(df.format(dan) + "*" +  df.format(i) + "=" + df.format(dan*i) + "\t");
			} // for i
			System.out.println();
		} // for dan				

		System.out.println();
		
		for(dan=2; dan<10; dan++); { // for-dan result: dan=10  
			for(i=1; i<10; i++) {
				System.out.print(df.format(dan) + "*" +  df.format(i) + "=" + df.format(dan*i) + "\t");
			} // for i
			System.out.println();
		} // meaningless
		
	}
}


/*
[문제] 2~9단
 */
