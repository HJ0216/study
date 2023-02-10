import java.util.*;

public class ComputeMain {

	public static void main(String[] args) {
		
		Scanner scan = new Scanner(System.in);

		System.out.print("횟수 입력: ");
		int input = scan.nextInt();

		int[] x = new int[input];
		int[] y = new int[input];		
		Compute[] cpt = new Compute[input];
		
		for(int i=0; i<input; i++) {
			System.out.println("["+ (i+1) + "번째]");
			System.out.print("x" + (i+1) + " 입력: ");
			x[i] = scan.nextInt();
			System.out.print("y" + (i+1) + " 입력: ");
			y[i] = scan.nextInt();
			
			System.out.println();
			
			cpt[i] = new Compute();
			cpt[i].setData(x[i], y[i]);

		}

		
		System.out.println("X\tY\tSUM\tSUB\tMUL\tDIV");
				
		for(int i=0; i<cpt.length; i++) {
			cpt[i].calc();
			System.out.println(cpt[i].getX()   + "\t"
							 + cpt[i].getY()   + "\t"
							 + cpt[i].getSum() + "\t"
							 + cpt[i].getSub() + "\t"
							 + cpt[i].getMul() + "\t"
							 + String.format("%.2f", cpt[i].getDiv()));
		}
//		for(compute data: cpt) {
		// Compute[] cpt = new Compute[input];
		// cpt는 Compute[]이므로 data의 data type은 Compute(Class type, Obj)
//			data.calc();
//			System.out.println(data.getX()   + "\t"
//							 + data.getY()   + "\t"
//							 + data.getSum() + "\t"
//							 + data.getSub() + "\t"
//							 + data.getMul() + "\t"
//							 + String.format("%.2f", data.getDiv()));
//			
//		}
		
		
		scan.close();
	}
}
